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
        mask_in_chans = 16
        embed_dim = d_model
        self.downscale1 = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.ReLU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.ReLU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

        self.downscale2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim),
            nn.ReLU()
        )

        self.downscale3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim),
            nn.ReLU()
        )

        self.downscale4 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim),
            nn.ReLU()
        )

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
        pandding_masks = []
        supp_mask=supp_masks.unsqueeze(1).float()
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = supp_features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
            if idx==0:
                supp_mask=self.downscale1(supp_mask)
            elif idx==1:
                supp_mask = self.downscale2(supp_mask)
            elif idx==2:
                supp_mask = self.downscale3(supp_mask)
            else:
                supp_mask = self.downscale4(supp_mask)
            padding_mask = F.interpolate(supp_masks.unsqueeze(1).float(), size=x.shape[-2:], mode='nearest').squeeze(1).to(torch.int64)
            masks.append(supp_mask)
            pandding_masks.append(padding_mask)

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, padding_mask,pos_embed) in enumerate(zip(srcs, masks[::-1],pandding_masks, pos)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # src=src+mask
            # src = src * mask
            src = src.flatten(2).transpose(1, 2)
            padding_mask = padding_mask.flatten(1).to(torch.bool)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(padding_mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(~m) for m in pandding_masks], 1)

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
                memory_key_padding_mask=None,#memory_key_padding_mask,
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



