import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import einsum
from einops import rearrange, repeat

from mmdet3d.ops.furthest_point_sample.furthest_point_sample import furthest_point_sample
from mmdet3d.ops import (PointFPModule, Points_Sampler, QueryAndGroup,
                         gather_points)
from mmcv.cnn import ConvModule
from mmdet.models import BACKBONES


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim * 3, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim

    def forward(self, pos_embed, max_len=(1, 1, 1)):
        z_embed, y_embed, x_embed = pos_embed.chunk(3, 1)
        z_max, y_max, x_max = max_len

        eps = 1e-6
        z_embed = z_embed / (z_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale
        x_embed = x_embed / (x_max + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_z = z_embed / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(),
                             pos_z[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=1)

        pos = self.token_projection(pos)
        return pos


class LocalGrouper(nn.Module):
    def __init__(self, num_point, radius, num_sample, use_xyz=True):
        super(LocalGrouper, self).__init__()
        self.num_point = num_point
        self.radius = radius
        self.num_sample = num_sample
        self.use_xyz = use_xyz
        self.sampler = Points_Sampler([self.num_point], ['D-FPS'])
        self.grouper = QueryAndGroup(self.radius, self.num_sample, use_xyz=use_xyz)

    def forward(self, xyz, features):
        xyz_flipped = xyz.transpose(1, 2).contiguous()  # B, 3, N
        fps_idx = self.sampler(xyz, features)
        new_xyz = gather_points(xyz_flipped, fps_idx).transpose(1, 2).contiguous()  # B, npoint, 3
        grouped_features = self.grouper(xyz, new_xyz, features)  # (B, 3 + C, npoint, num_sample)
        return new_xyz, grouped_features, fps_idx


class DynamicInteraction(nn.Module):
    def __init__(self, channels, reduce, num_sample):

        super(DynamicInteraction, self).__init__()
        self.channels = channels
        self.reduce = reduce
        self.num_sample = num_sample
        self.self_attn = nn.MultiheadAttention(channels, num_heads=4, dropout=0.0)
        self.norm_attn = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.0, inplace=True)
        self.net1 = ConvModule(in_channels=channels, out_channels=int(2 * channels * reduce * channels),
                              kernel_size=1, conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'))
        self.relu1 = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(int(channels * reduce))
        self.net2 = ConvModule(in_channels=channels * num_sample, out_channels=channels,
                               kernel_size=1, conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'))
        self.relu2 = nn.ReLU(inplace=True)
        self.norm2 = nn.LayerNorm(channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, pre_features, features):
        B, C, N = features.shape
        # features1 = features.permute(0, 2, 1)
        # features2 = self.self_attn(features1, features1, features1)[0]
        # features2 = self.dropout(features2)
        # features3 = self.norm_attn(features1 + features2)
        # features = features3.permute(0, 2, 1)
        dynamic_features = self.net1(features).permute(0, 2, 1).contiguous()  # (B, N, 2*C*C/4)
        param1 = dynamic_features[:, :, :int(self.channels * self.reduce * self.channels)]\
            .view(B, N, self.channels, int(self.channels * self.reduce))  # (B, N, C, C/4)
        param2 = dynamic_features[:, :, int(self.channels * self.reduce * self.channels):] \
            .view(B, N, int(self.channels * self.reduce), self.channels)  # (B, N, C/4, C)
        pre_features = pre_features.permute(0, 2, 3, 1)  # B, N, S, C
        tmp_features = self.relu1(self.norm1(torch.matmul(pre_features, param1)))
        tmp_features = self.relu2(self.norm2(torch.matmul(tmp_features, param2)))
        tmp_features = torch.flatten(tmp_features, 2)
        tmp_features = tmp_features.permute(0, 2, 1).contiguous()
        features = self.relu3(self.net2(tmp_features) + features)
        return features


class ConvResModule(nn.Module):
    def __init__(self, channel, res_expansion=1.0):
        super(ConvResModule, self).__init__()
        self.net1 = ConvModule(in_channels=channel, out_channels=int(channel * res_expansion), kernel_size=1,
                               conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'))
        self.net2 = ConvModule(in_channels=int(channel * res_expansion), out_channels=channel, kernel_size=1,
                               conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'), act_cfg=None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        tmp_features = self.net1(features)
        tmp_features = self.net2(tmp_features)
        final_features = self.relu(tmp_features + features)
        return final_features


class FeaExtraction(nn.Module):
    def __init__(self, channels, out_channels, num_sample, blocks, res_espension, use_xyz=True, use_pos_emb=True):
        super(FeaExtraction, self).__init__()
        in_channels = 3 + channels if use_xyz else channels
        self.num_sample = num_sample
        self.use_pos_emb = use_pos_emb
        self.transfer = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'))
        if use_pos_emb:
            self.pe = PositionalEncodingFourier(int(channels // 2), channels)
            self.trans = ConvModule(in_channels=channels, out_channels=out_channels, kernel_size=1,
                                   conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'))

        operation = []
        for _ in range(blocks):
            operation.append(
                ConvResModule(out_channels, res_espension)
            )
        self.operation = nn.Sequential(*operation)
        self.dy_interaction = DynamicInteraction(out_channels, 0.25, self.num_sample)

    def forward(self, features):
        if self.use_pos_emb:
            b, c, n, s = features.size()
            xyz = features[:, :3, :, :]
            features = features[:, 3:, :, :]
            xyz = xyz.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            pe = self.pe(xyz)
            pe = pe.view(b, n, s, -1).permute(0, 3, 1, 2).contiguous()
            features = features + pe
            features = self.trans(features)
        else:
            features = self.transfer(features)
        ori_features = features
        b, c, n, s = features.size()
        features = features.permute(0, 2, 1, 3).contiguous().view(-1, c, s)
        batch_size, _, _ = features.size()
        features = self.operation(features)
        features = F.adaptive_max_pool1d(features, 1).view(batch_size, -1).contiguous()
        features = features.view(b, n, -1).permute(0, 2, 1).contiguous()
        final_features = self.dy_interaction(ori_features, features)
        return final_features


@BACKBONES.register_module()
class DynamicPointInteraction(nn.Module):
    def __init__(self,
                 num_point=(2048, 1024, 512, 256),
                 radii=(0.2, 0.4, 0.8, 1.2),
                 num_sample=(64, 32, 16, 16),
                 embed_dim=64,
                 gmp_dim=64,
                 res_expansion=1.0,
                 use_xyz=True,
                 use_pos_emb=True,
                 ed_dims=(128, 256, 512, 512),
                 fea_blocks=(2, 2, 2, 2),
                 fp_dims=(256, 256)):
        super(DynamicPointInteraction, self).__init__()
        self.stages = len(fea_blocks)
        self.num_point = num_point
        self.radii =radii
        self.num_sample = num_sample
        self.ed_dims = ed_dims
        self.fea_blocks = fea_blocks
        self.fp_dims = fp_dims
        self.embedding = ConvModule(in_channels=1, out_channels=embed_dim, kernel_size=1,
                                    conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'))
        self.local_grouper_list = nn.ModuleList()
        self.fea_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        en_dims = [last_channel]
        # build encoder
        for i in range(len(num_point)):
            local_grouper = LocalGrouper(num_point=self.num_point[i], radius=self.radii[i],
                                         num_sample=self.num_sample[i], use_xyz=use_xyz)
            self.local_grouper_list.append(local_grouper)
            fea_block_module = FeaExtraction(last_channel, self.ed_dims[i], self.num_sample[i], self.fea_blocks[i],
                                             res_espension=res_expansion, use_xyz=use_xyz, use_pos_emb=use_pos_emb)
            self.fea_blocks_list.append(fea_block_module)
            last_channel = self.ed_dims[i]
            en_dims.append(last_channel)

        # build decoder
        self.decode_list = nn.ModuleList()
        self.decode_list.append(
            PointFPModule([en_dims[-1] + en_dims[-2], fp_dims[0], fp_dims[0]], norm_cfg=dict(type='BN2d'))
        )
        self.decode_list.append(
            PointFPModule([en_dims[-3] + fp_dims[0], fp_dims[1], fp_dims[1]], norm_cfg=dict(type='BN2d'))
        )
        self.num_fp = len(self.decode_list)

        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for ed_dim in ed_dims:
            self.gmp_map_list.append(ConvModule(ed_dim, gmp_dim, kernel_size=1,
                                                conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d')))
        self.gmp_map_end = ConvModule(64 * len(ed_dims), gmp_dim, kernel_size=1,
                                      conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'))
        self.gmp_embedding = ConvModule(in_channels=gmp_dim + fp_dims[1], out_channels=fp_dims[1], kernel_size=1,
                                        conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'))

    def forward(self, points):
        xyz = points[..., 0:3].contiguous()
        features = self.embedding(points[..., 3:].transpose(1, 2))
        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            cur_xyz, cur_features, cur_indices = self.local_grouper_list[i](
                sa_xyz[i], sa_features[i])  # [b,g,3]  [b,g,k,d]
            cur_features = self.fea_blocks_list[i](cur_features)  # [b,d,g]
            sa_xyz.append(cur_xyz.contiguous())
            sa_features.append(cur_features.contiguous())
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        # here is the decoder
        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]
        fp_indices = [sa_indices[-1]]
        for i in range(self.num_fp):
            fp_features.append(self.decode_list[i](
                sa_xyz[self.stages - i - 1], sa_xyz[self.stages - i],
                sa_features[self.stages - i - 1], fp_features[-1]))
            fp_xyz.append(sa_xyz[self.stages - i - 1])
            fp_indices.append(sa_indices[self.stages - i - 1])

        # here is the global context
        gmp_list = []
        for i in range(self.stages):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](sa_features[i+1]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1))  # [b, gmp_dim, 1]

        features = fp_features[-1]
        final_features = torch.cat([features, global_context.repeat(1, 1, features.shape[-1])], dim=1)
        final_features = self.gmp_embedding(final_features)
        fp_features[-1] = final_features

        ret = dict(fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)
        return ret
