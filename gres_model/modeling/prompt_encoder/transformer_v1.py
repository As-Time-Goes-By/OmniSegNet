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
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.refpoint_embed = nn.Embedding(num_prompts, 4)

    def forward(
        self,
        supp_features,
        supp_masks,
        target
        # memory,
        # tgt_mask: Optional[Tensor] = None,
        # memory_mask: Optional[Tensor] = None,
        # tgt_key_padding_mask: Optional[Tensor] = None,
        # memory_key_padding_mask: Optional[Tensor] = None,
        # pos: Optional[Tensor] = None,
        # refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        # level_start_index: Optional[Tensor] = None,  # num_levels
        # spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        # valid_ratios: Optional[Tensor] = None,
        # for text
        # memory_text: Optional[Tensor] = None,
        # text_attention_mask: Optional[Tensor] = None,
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
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        batch_size=src_flatten.shape[0]
        output=target
        # output_pos=self.pe_layer(output)
        # output = self.vis_prompts.weight.unsqueeze(0).expand(batch_size, -1, -1)

        memory=src_flatten
        # memory_key_padding_mask=mask_flatten
        memory_key_padding_mask= ~mask_flatten

        intermediate = []
        reference_points = self.refpoint_embed.weight[:, None, :].repeat(1, batch_size, 1).sigmoid()
        ref_points = [reference_points]

        # print('valid_ratios',valid_ratios.shape)##[4,4,2]
        # exit(0)
        for layer_id, layer in enumerate(self.layers):


            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()
            # print('query_pos shape',query_pos.shape)
            # main process

            # print('output',output.shape,'memory',memory.shape,'lvl_pos_embed_flatten',lvl_pos_embed_flatten.shape)
            # exit(0)
            output = layer(
                tgt=output,
                tgt_query_pos=None,
                memory=memory,
                memory_pos=lvl_pos_embed_flatten,
                self_attn_mask=None,
                cross_attn_mask=None
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


            output=self.norm(output)
            intermediate.append(output)

        # import pdb;pdb.set_trace()

        return output

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio





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
        # self.cross_attn = MSDeformAttn(
        #     embed_dim=d_model,
        #     num_levels=n_levels,
        #     num_heads=n_heads,
        #     num_points=n_points,
        #     batch_first=True,
        # )
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=True)

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        # if use_text_cross_attention:
        #     self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        #     self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        #     self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=True)
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

        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_pos: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        # assert cross_attn_mask is None

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
            self.with_pos_embed(tgt, tgt_query_pos),
            self.with_pos_embed(memory, memory_pos),
            self.with_pos_embed(memory, memory_pos),
            attn_mask=cross_attn_mask
        )[0]
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



