# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn

from .utils import inverse_sigmoid
import torch.nn.functional as F
from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .transformer_vanilla import TransformerEncoderLayer
from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)
from ..transformer_decoder.position_encoding import PositionEmbeddingSine

class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        # for deformable encoder
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        # init query
        learnable_tgt_init=False,
        # two stage
        two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
        embed_init_tgt=False,
        # for text
        use_text_enhancer=False,
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        text_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4

        # choose encoder layer type
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )

        if use_text_enhancer:
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout,
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer:
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # choose decoder layer type
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, text_dict=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################
        text_dict["encoded_text"] = memory_text
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if memory.isnan().any() | memory.isinf().any():
        #         import ipdb; ipdb.set_trace()

        if self.two_stage_type == "standard":  #把encoder的输出作为proposal
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if text_dict is not None:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict)
            else:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

            topk_logits = enc_outputs_class_unselected.max(-1)[0]
            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries

            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
            )
            if self.embed_init_tgt:
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == "no":
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################

        #memory  torch.Size([2, 16320, 256])

        # import pdb;pdb.set_trace()
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == "standard":
            hs_enc = tgt_undetach.unsqueeze(0)
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None





class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        input_shape,
        transformer_in_features,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=4,

    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model

        self.ref_anchor_head = None

        N_steps = d_model // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        # print(input_shape)
        # print(transformer_in_features)
        # exit(0)
        transformer_in_channels = [v.channels for k, v in transformer_input_shape.items()]
        self.transformer_num_feature_levels = len(transformer_in_features)
        self.transformer_in_features=transformer_in_features
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)


        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_prompts=20
        # self.vis_prompts = nn.Embedding(num_prompts, d_model)
        self.vis_prompts_pos = nn.Embedding(num_prompts, d_model)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.refpoint_embed = nn.Embedding(num_prompts, 4)
        self.reference_points = nn.Parameter(torch.rand(num_prompts, 4, 2))

    def forward(
        self,
        supp_features,
        supp_masks,
        target

    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        srcs = []
        pos = []
        masks=[]
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = supp_features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
            supp_mask = F.interpolate(supp_masks.unsqueeze(1).float(), size=x.shape[-2:], mode='nearest').squeeze(1).to(torch.int64)
            masks.append(supp_mask)

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1).to(torch.bool)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(~m) for m in masks], 1)

        batch_size=src_flatten.shape[0]
        output=target
        # output = self.vis_prompts.weight.unsqueeze(0).expand(batch_size, -1, -1)
        output_pos = self.vis_prompts_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        memory=src_flatten
        # memory_key_padding_mask=mask_flatten
        memory_key_padding_mask= ~mask_flatten

        intermediate = []
        # reference_points = self.refpoint_embed.weight[:, None, :].repeat(1, batch_size, 1).sigmoid()
        # ref_points = [reference_points]

        # query_e = int(math.sqrt(query.shape[1]))
        reference_points = self.get_reference_points([(4, 5)], device=output.device)

        # print('valid_ratios',valid_ratios.shape)##[4,4,2]
        # print('reference_points', reference_points.shape)
        # exit(0)
        for layer_id, layer in enumerate(self.layers):

            # if reference_points.shape[-1] == 4:
            #     reference_points_input = (
            #         reference_points[:, :, None]
            #         * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            #     )  # nq, bs, nlevel, 4
            # else:
            #     assert reference_points.shape[-1] == 2
            #     reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            # query_sine_embed = gen_sineembed_for_position(
            #     reference_points_input[:, :, 0, :]
            # )  # nq, bs, 256*2
            #
            # # conditional query
            # raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            # pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            # query_pos = pos_scale * raw_query_pos


            output = layer(
                tgt=output,
                tgt_query_pos=output_pos,#query_pos.transpose(0, 1),
                tgt_query_sine_embed=None,
                tgt_key_padding_mask=None,
                tgt_reference_points=reference_points,
                memory_text=None,
                text_attention_mask=None,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=lvl_pos_embed_flatten,
                self_attn_mask=None,
                cross_attn_mask=None,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # iter update
            # if self.bbox_embed is not None:
            #     # box_holder = self.bbox_embed(output)
            #     # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
            #     # new_reference_points = box_holder[..., :self.query_dim].sigmoid()
            #
            #     reference_before_sigmoid = inverse_sigmoid(reference_points)
            #     delta_unsig = self.bbox_embed[layer_id](output)
            #     outputs_unsig = delta_unsig + reference_before_sigmoid
            #     new_reference_points = outputs_unsig.sigmoid()
            #
            #     reference_points = new_reference_points.detach()
            #     # if layer_id != self.num_layers - 1:
            #     ref_points.append(new_reference_points)

            # output=self.norm(output)
            intermediate.append(self.norm(output))

        # import pdb;pdb.set_trace()

        return intermediate[-1]

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self,spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points





class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        # if use_text_cross_attention:
        #     self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        #     self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        #     self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        # if self.self_attn is not None:
        #     # import ipdb; ipdb.set_trace()
        #     q = k = self.with_pos_embed(tgt, tgt_query_pos)
        #     tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        #     tgt = tgt + self.dropout2(tgt2)
        #     tgt = self.norm2(tgt)


        # print('tgt',tgt.shape,'tgt_reference_points',tgt_reference_points.shape,'memory',memory.shape)
        # exit(0)
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            reference_points=tgt_reference_points,#.transpose(0, 1).contiguous(),
            value=memory,
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        embed_init_tgt=args.embed_init_tgt,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
    )
