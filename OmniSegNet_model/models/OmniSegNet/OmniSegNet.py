from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from transformers import BertModel

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from OmniSegNet_model.modeling.criterion import ReferringCriterion
from OmniSegNet_model.modeling.meta_arch.referring_head import ReferringHead
from OmniSegNet_model.modeling.prompt_encoder.prompt import get_scribble_mask, get_bounding_boxes

from OmniSegNet_model.modeling.backbone.swin import D2SwinTransformer
from torchvision.ops import masks_to_boxes, roi_align
from OmniSegNet_model.modeling.prompt_encoder.transformer import TransformerDecoder,DeformableTransformerDecoderLayer


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class OmniSegNet(nn.Module):

    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        lang_backbone: nn.Module,
        condition: str
        # # transform_decoder: nn.Module,
    ):

        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
       
        self.ref_criterion = criterion
        self.supp_criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # language backbone
        self.text_encoder = lang_backbone
        self.in_channels = [128 * (2 ** i) for i in range(4)]
        self.feat_feature = 256
        self.lang_feature = 768
        self.feature_map_proj = nn.Conv2d(sum(self.in_channels), self.feat_feature, kernel_size=1)
        self.lang_in_linear = nn.Linear(self.feat_feature, self.lang_feature)
        self.lang_in_norm = nn.LayerNorm(self.lang_feature)
        self.lang_norm = nn.LayerNorm(self.lang_feature)
        self.feat_feature = 256
        decoder_norm = nn.LayerNorm(self.feat_feature)
        prompt_encoder_layer = DeformableTransformerDecoderLayer(d_model=self.feat_feature, d_ffn=2048, dropout=0.0)
        self.transformer_in_features = ["res2", "res3", "res4", "res5"]
        input_shape = {
            k: v for k, v in backbone.output_shape().items() if k in self.transformer_in_features
        }

        self.prompt_encoder = TransformerDecoder(prompt_encoder_layer, 3, input_shape, self.transformer_in_features,
                                            norm=decoder_norm,
                                            return_intermediate=True,
                                            d_model=self.feat_feature,
                                            query_dim=4,
                                            num_feature_levels=4, )
        self.num_prompts = 20
        self.mask_in_chans=256
        self.vis_prompts = nn.Embedding(self.num_prompts, self.feat_feature)
        self.condition = condition
    @classmethod
    def from_config(cls, cfg):
      
        backbone =D2SwinTransformer(cfg)
      
        sem_seg_head_components = ReferringHead.from_config(cfg, backbone.output_shape())
        if cfg.MODEL.SWIN.SWIN_PRETRAINED_WEIGHTS:
            print('Initializing Multi-modal Swin Transformer weights from ' + cfg.MODEL.SWIN.SWIN_PRETRAINED_WEIGHTS)
            backbone.init_weights(pretrained=cfg.MODEL.SWIN.SWIN_PRETRAINED_WEIGHTS)
        sem_seg_head = ReferringHead(**sem_seg_head_components)
        text_encoder = BertModel.from_pretrained(cfg.REFERRING.BERT_TYPE)
        text_encoder.pooler = None

        ###prompt encoder


        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        losses = ["masks"]

        criterion = ReferringCriterion(
            weight_dict=weight_dict,
            losses=losses,
        )

        return {

            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "lang_backbone": text_encoder,
            "condition": cfg.CONDITION

        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, ref_data={},supp_data={}):
       
        if supp_data.get('image', None) is not None and ref_data.get('image', None) is not None:
            images = torch.cat([ref_data['image'], supp_data['image']], dim=0)
        elif supp_data.get('image', None) is not None:
            images = supp_data['image']
        elif ref_data.get('image', None) is not None:
            images = ref_data['image']
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
       
        if ref_data.get('lang_tokens', None) is not None:
            lang_emb = ref_data['lang_tokens'].squeeze(1)
            lang_mask = ref_data['lang_mask'].squeeze(1)
            lang_feat = self.text_encoder(lang_emb, attention_mask=lang_mask)[0]  # B, Nl, 768
            lang_feat = self.lang_norm(lang_feat)
            ref_feat = lang_feat.permute(0, 2, 1)  
            ref_mask = lang_mask.unsqueeze(dim=-1)  
        else:
            ref_feat = None  
            ref_mask = None  

        if supp_data.get('supp_image', None) is not None:
            supp_images = supp_data['supp_image']
            supp_masks = supp_data['supp_mask']
           
            if self.condition == 'scribble':
                supp_masks = get_scribble_mask(supp_masks, self.training)  # scribble_mask
            elif self.condition == 'box':
                boxes, supp_masks = get_bounding_boxes(supp_masks)  # box_mask
            elif self.condition == 'mask':
                supp_masks = supp_masks

            supp_features = self.backbone(supp_images, None, None)

            target = self.vis_prompts.weight.unsqueeze(0).repeat(supp_images.shape[0], 1, 1)
       

            visual_tokens=self.prompt_encoder(supp_features,supp_masks,target)

            visual_masks = torch.ones(visual_tokens.shape[0], visual_tokens.shape[1]).to(self.device)

            visual_tokens = self.lang_in_norm(self.lang_in_linear(visual_tokens))  
           
            supp_feat = visual_tokens.permute(0, 2, 1) 
            supp_mask = visual_masks.unsqueeze(dim=-1)  
        else:
            supp_feat = None  
            supp_mask = None  

        if ref_feat is not None and supp_feat is not None:
            all_feat = torch.cat([ref_feat, supp_feat], dim=0)
            all_mask = torch.cat([ref_mask, supp_mask], dim=0)
        elif ref_feat is not None:
            all_feat = ref_feat
            all_mask = ref_mask
        elif supp_feat is not None:
            all_feat = supp_feat
            all_mask = supp_mask

           
        features = self.backbone(images.tensor, all_feat, all_mask)
        outputs = self.sem_seg_head(features, all_feat, all_mask)

        if self.training:
           
            if ref_feat is not None:
                ref_targets = self.prepare_targets({
                    "gt_mask_merged": ref_data['gt_mask_merged'],
                    "empty": ref_data['empty']
                }, ref_data['image'])
            if supp_feat is not None:
                supp_targets = self.prepare_targets({
                    "gt_mask_merged": supp_data['gt_mask_merged'],
                    "empty": supp_data['empty']
                }, supp_data['image'])

           
            if ref_feat is not None:
                
                ref_outputs = {}  
                ref_batch = ref_data["gt_mask_merged"].shape[0]
                for key, value in outputs.items():
                    ref_outputs[key] = value[:ref_batch]
                ref_losses = self.ref_criterion(ref_outputs, ref_targets)

            if supp_feat is not None:
                
                supp_outputs = {} 
                supp_batch = supp_data["gt_mask_merged"].shape[0]
                for key, value in outputs.items():
                    supp_outputs[key] = value[-supp_batch:]
                supp_losses = self.supp_criterion(supp_outputs, supp_targets)

            if ref_feat is not None:
                for k in list(ref_losses.keys()):
                    if k in self.ref_criterion.weight_dict:
                        ref_losses[k] *= self.ref_criterion.weight_dict[k]
                    else:
                        ref_losses.pop(k)
            if supp_feat is not None:
                for k in list(supp_losses.keys()):
                    if k in self.supp_criterion.weight_dict:
                        supp_losses[k] *= self.supp_criterion.weight_dict[k]
                    else:
                        supp_losses.pop(k)
            losses = {'loss_mask': supp_losses['loss_mask'],
                          'loss': supp_losses['loss_mask']}
            # or
            # losses = {'loss_mask': ref_losses['loss_mask'],
            #           'loss': ref_losses['loss_mask']}


          

            return losses
        else:
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            nt_pred_results = outputs["nt_label"]

            del outputs

            processed_results = []
            for mask_pred_result, nt_pred_result,  image_size in zip(
                mask_pred_results, nt_pred_results,  images.image_sizes
            ):
                processed_results.append({})
                r, nt = retry_if_cuda_oom(self.refer_inference)(mask_pred_result, nt_pred_result)
                processed_results[-1]["ref_seg"] = r
                processed_results[-1]["nt_label"] = nt

            return processed_results


    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.shape[-2:]
        batch_size=images.shape[0]
        new_targets = []

        for i in range(batch_size):

            is_empty = batched_inputs['empty'][i].clone().detach().to(dtype=torch.int64)

            target_dict = {

                    "empty": is_empty,
                }
            if batched_inputs["gt_mask_merged"] is not None:
                target_dict["gt_mask_merged"] = batched_inputs["gt_mask_merged"][i].to(self.device)

            new_targets.append(target_dict)
        return new_targets

    
    def refer_inference(self, mask_pred, nt_pred):
        mask_pred = mask_pred.sigmoid()
        nt_pred = nt_pred.sigmoid()
        return mask_pred, nt_pred

    