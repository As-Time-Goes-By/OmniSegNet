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

from gres_model.modeling.criterion import ReferringCriterion
from gres_model.modeling.meta_arch.referring_head import ReferringHead
from gres_model.modeling.prompt_encoder.prompt import get_scribble_mask, get_point_mask, get_bounding_boxes,get_bounding_boxes_v1

from gres_model.modeling.backbone.swin import D2SwinTransformer
from torchvision.ops import masks_to_boxes, roi_align
from gres_model.modeling.prompt_encoder.transformer_v3 import TransformerDecoder,DeformableTransformerDecoderLayer
# from gres_model.modeling.prompt_encoder.transformer_v1 import TransformerDecoder,DeformableTransformerDecoderLayer
# @META_ARCH_REGISTRY.register()

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
class GRES(nn.Module):
    # @configurable
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
        self.criterion = criterion
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
        # backbone = build_backbone(cfg)
        backbone =D2SwinTransformer(cfg)
        # backbone=backbone.init_weights(pretrained=cfg.MODEL.SWIN.SWIN_PRETRAINED_WEIGHTS)
        # print(backbone.output_shape())
        # exit(0)
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
            # 'transform_decoder': prompt_encoder
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # print('*'*20,type(batched_inputs))
        # images = [x["image"].to(self.device) for x in batched_inputs]
        images=batched_inputs['image']
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        # images=torch.stack(images)
        # print(batched_inputs.keys())
        # exit(0)
        if batched_inputs.get('lang_tokens', None) is not None:
            lang_emb=batched_inputs['lang_tokens'].squeeze(1)
            lang_mask = batched_inputs['lang_mask'].squeeze(1)
            lang_feat = self.text_encoder(lang_emb, attention_mask=lang_mask)[0] # B, Nl, 768
            lang_feat = self.lang_norm(lang_feat)
            ref_feat = lang_feat.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            ref_mask = lang_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        elif batched_inputs.get('supp_image', None) is not None:
            supp_images = batched_inputs['supp_image']
            supp_masks=batched_inputs['supp_mask']
            # supp_masks = supp_masks.unsqueeze(1)  # 从 [B, H, W] 扩展到 [B, 1, H, W]
            # un_supp_masks = un_supp_masks.expand(-1, supp_images.shape[1], -1, -1)
            # supp_images = supp_images * un_supp_masks
            if self.condition == 'scribble':
                supp_masks = get_scribble_mask(supp_masks, training=self.training)  # scribble_mask
            elif self.condition == 'point':
                supp_masks = get_point_mask(supp_masks, training=self.training,max_points=20)  # point_mask
            elif self.condition == 'box':
                boxes, supp_masks = get_bounding_boxes(supp_masks)  # box_mask
                # boxes, supp_masks = get_bounding_boxes_v1(supp_masks)  # box_mask
            elif self.condition == 'mask':
                supp_masks = supp_masks

            supp_features = self.backbone(supp_images, None, None)

            target = self.vis_prompts.weight.unsqueeze(0).repeat(supp_images.shape[0], 1, 1)
            # target =self.mask_downscaling(supp_masks.unsqueeze(1).float())

            visual_tokens=self.prompt_encoder(supp_features,supp_masks,target)

            # gate_weights = torch.sigmoid(target)
            # visual_tokens=visual_tokens.permute(0, 2, 1).reshape(-1, 256, 30, 30)
            # visual_tokens=F.adaptive_avg_pool2d(visual_tokens, (4,5))

            # combined_features = self.combine_features(supp_features, supp_masks)  # 融合多尺度特征
            # print('visual_tokens shape', visual_tokens.shape)
            # exit(0)
            # visual_tokens = visual_tokens.flatten(2).permute(0, 2, 1)  # (B, N_l, C)
            visual_masks = torch.ones(visual_tokens.shape[0], visual_tokens.shape[1]).to(self.device)

            visual_tokens = self.lang_in_norm(self.lang_in_linear(visual_tokens))  # [B, N_l, C]
            # visual_tokens = visual_tokens.permute(0, 2, 1) #(B, 768, N_l)
            # visual_masks = visual_masks.unsqueeze(dim=-1)   #(batch, N_l, 1)
            ref_feat = visual_tokens.permute(0, 2, 1)  # (B, 768, N_l)
            ref_mask = visual_masks.unsqueeze(dim=-1)  # (batch, N_l, 1)

            # features = self.backbone(images.tensor, None, None)
        features = self.backbone(images.tensor, ref_feat, ref_mask)
        outputs = self.sem_seg_head(features, ref_feat, ref_mask)

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)

            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
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
        h_pad, w_pad = images.tensor.shape[-2:]
        batch_size=images.tensor.shape[0]
        new_targets = []

        for i in range(batch_size):
            # pad instances
            # targets_per_image = data_per_image['instances'].to(self.device)
            # gt_masks = targets_per_image.gt_masks
            # padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            # padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            # padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)

    #         is_empty = batched_inputs['empty'][i, 0].clone().detach().to(dtype=torch.int64)
    #         is_empty = torch.tensor(batched_inputs['empty'][i], dtype=torch.int64
    #         , device=batched_inputs["empty"].device)
            is_empty = batched_inputs['empty'][i].clone().detach().to(dtype=torch.int64)

            target_dict = {
                    # "labels": batched_inputs["gt_classes"][i,0],
                    # "masks": padded_masks,
                    "empty": is_empty,
                }
            if batched_inputs["gt_mask_merged"] is not None:
                target_dict["gt_mask_merged"] = batched_inputs["gt_mask_merged"][i].to(self.device)

            new_targets.append(target_dict)
        return new_targets

    def prepare_targets_v2(self, batched_inputs, images):
        """
        Prepares target tensors for a batch of inputs.

        Args:
            batched_inputs (list[dict]): List of dictionaries, each containing input data and annotations.
            images (torch.Tensor): Tensor of batched images with padding applied, of shape [B, C, H_pad, W_pad].

        Returns:
            list[dict]: List of dictionaries containing processed targets.
        """
        h_pad, w_pad = images.shape[-2:]
        new_targets = []

        for data_per_image in batched_inputs:
            # Extract ground truth masks
            gt_masks = torch.stack(data_per_image['gt_masks'],dim=0) # Tensor of shape [N, H, W]
            gt_masks = gt_masks.to(self.device)

            # Pad masks to match the padded image size
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device
            )
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks

            # Handle empty instances
            is_empty = torch.tensor(
                data_per_image.get('empty', False),  # Default to False if 'empty' is not provided
                dtype=data_per_image['gt_classes'].dtype,
                device=data_per_image['gt_classes'].device
            )

            # Create target dictionary
            target_dict = {
                "labels": data_per_image['gt_classes'],  # Ground truth classes
                "masks": padded_masks,
                "empty": is_empty,
            }

            # Optional merged mask
            if data_per_image.get("gt_mask_merged") is not None:
                target_dict["gt_mask_merged"] = data_per_image["gt_mask_merged"].to(self.device)

            new_targets.append(target_dict)

        return new_targets
    def refer_inference(self, mask_pred, nt_pred):
        mask_pred = mask_pred.sigmoid()
        nt_pred = nt_pred.sigmoid()
        return mask_pred, nt_pred

    def combine_features(self,supp_features,supp_masks):
        h,w=supp_features['res3'].shape[-2:]
        supp_mask = F.interpolate(supp_masks.unsqueeze(1).float(), size=(h,w), mode='nearest')
        feat_list=[]
        for feat in supp_features.values():
            feat=F.interpolate(feat.float(), size=(h,w), mode='bilinear',align_corners=True)
            # feat=feat*supp_mask
            feat_list.append(feat)

        final_feature=torch.cat(feat_list,dim=1)

        final_feature=self.feature_map_proj(final_feature)
        return final_feature

    def prepare_boxes(self,masks):
        # masks: Tensor [B, 1, H, W] (binary masks)
        bs = masks.size(0)  # batch size
        boxes = masks_to_boxes(masks)  # [B, K, 4]
        batch_indices = torch.arange(bs, device=masks.device).repeat_interleave(boxes.size(0) // bs)
        boxes = torch.cat([batch_indices.unsqueeze(1), boxes], dim=1)  # [K, 5]
        # print(boxes)
        return boxes

