import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
from models.arch_ciaosr.arch_csnln import CrossScaleAttention


@register('lmciaosr')
class LMCiaoSR(nn.Module):
    def __init__(self, encoder, prenet_q, hypernet_q,
                 imnet_q, imnet_k, imnet_v,
                 local_size=2, feat_unfold=True, non_local_attn=True,
                 multi_scale=[2], softmax_scale=1, mod_input=False,
                 cmsr_spec=None):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.unfold_range = 3 if feat_unfold else 1
        self.unfold_dim = self.unfold_range * self.unfold_range
        # self.eval_bsize = eval_bsize
        self.local_size = local_size
        self.non_local_attn = non_local_attn
        self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale
        self.max_scale = 4  # Max training scale
        self.mod_input = mod_input  # Set to True if use compressed latent code

        self.encoder = models.make(encoder)

        if self.non_local_attn:
            self.non_local_attn_dim = self.encoder.out_dim * len(multi_scale)
            self.cs_attn = CrossScaleAttention(channel=self.encoder.out_dim, scale=multi_scale)

        # k&v imnet
        imnet_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_dim *= 2 * 2  # self.unfold_dim
        imnet_k_in_dim = imnet_v_in_dim = imnet_dim
        imnet_k_out_dim = imnet_v_out_dim = imnet_dim
        # coord + cell decoding
        imnet_k_in_dim += 4
        imnet_v_in_dim += 4

        real_unfold_dim = 2 * 2
        if self.non_local_attn:
            imnet_v_in_dim += self.non_local_attn_dim
            imnet_v_out_dim += self.non_local_attn_dim

        self.imnet_k = models.make(imnet_k, args={'in_dim': imnet_k_in_dim, 'out_dim': imnet_k_out_dim})
        self.imnet_v = models.make(imnet_v, args={'in_dim': imnet_v_in_dim, 'out_dim': imnet_v_out_dim})

        # Use latent MLPs to generate modulations for the render MLP
        hypernet_q_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            hypernet_q_in_dim *= real_unfold_dim  #self.unfold_dim
        if self.non_local_attn:
            hypernet_q_in_dim += self.non_local_attn_dim

        self.mod_dim = 0
        self.mod_dim += imnet_q['args']['hidden_dim'] if imnet_q['args']['mod_scale'] else 0
        self.mod_dim += imnet_q['args']['hidden_dim'] if imnet_q['args']['mod_shift'] else 0
        self.mod_dim *= imnet_q['args']['hidden_depth']
        hypernet_q_out_dim = self.mod_dim
        if self.mod_input:
            hypernet_q_out_dim += imnet_q['args']['hidden_dim'] * 2
        self.hypernet_q = models.make(hypernet_q, args={'in_dim': hypernet_q_in_dim, 'out_dim': hypernet_q_out_dim})

        # Convert 576-dim query to 256-dim query
        prenet_q_in_dim = self.encoder.out_dim * self.unfold_dim
        prenet_q_out_dim = self.encoder.out_dim * (real_unfold_dim if self.feat_unfold else 1)
        self.prenet_q = models.make(prenet_q, args={'in_dim': prenet_q_in_dim,
                                                    'out_dim': prenet_q_out_dim})

        imnet_q_in_dim = imnet_q['args']['hidden_dim'] * 2 if self.mod_input else imnet_dim
        imnet_q_in_dim += 2
        if self.non_local_attn and not self.mod_input:
            imnet_q_in_dim += self.non_local_attn_dim
        self.imnet_q = models.make(imnet_q, args={'in_dim': imnet_q_in_dim})

        # For time evaluation
        self.t_total = []
        self.feat_coord = None

        # Use CMSR in testing
        self.cmsr = cmsr_spec is not None
        if self.cmsr:
            self.mse_threshold = cmsr_spec["mse_threshold"]  # 0.00002
            self.s2m_tables = cmsr_spec["s2m_tables"]
            self.updating_cmsr = "updating_scale" in cmsr_spec

            if self.updating_cmsr:
                self.updating_scale = cmsr_spec["updating_scale"]
                self.scale2mean = self.s2m_tables[self.updating_scale]
                self.loss_fn = nn.MSELoss()
                print(f'Generating S2M Table at scale {self.updating_scale} created with MSE: {self.mse_threshold}')
            else:
                # Monitor the computational cost saved by CMSR
                self.cmsr_log = cmsr_spec["log"]
                if self.cmsr_log:
                    self.total_qn, self.total_q = 0, 0
                print(f'Using S2M Table created with MSE: {self.mse_threshold}')

    def gen_feats(self, inp, inp_coord=None):
        """
        Generate latent codes using the encoder.

        :param inp: Input image (B, h * w, 3)
        :param inp_coord: Input coordinates (B, h * w, 2)
        :return: Feature maps (B, C, h, w) and (B, 9*C, h, w)
        """

        feat = self.encoder(inp)
        [bs, in_c, in_h, in_w] = feat.shape

        if inp_coord is not None:
            self.feat_coord = inp_coord.permute(0, 3, 1, 2)
        elif self.training or self.feat_coord is None or self.feat_coord.shape[-2] != feat.shape[-2]\
                or self.feat_coord.shape[-1] != feat.shape[-1]:
            self.feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
                .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # self.t1 = time.time()

        if self.non_local_attn:
            crop_h, crop_w = 48, 48
            if in_h * in_w > crop_h * crop_w:
                # Fixme: Generate cross attention by image patches to avoid OOM
                self.non_local_feat = torch.zeros(bs, self.non_local_attn_dim, in_h, in_w).cuda()
                for i in range(in_h // crop_h):
                    for j in range(in_w // crop_w):
                        i1, i2 = i * crop_h, ((i + 1) * crop_h if i < in_h // crop_h - 1 else in_h)
                        j1, j2 = j * crop_w, ((j + 1) * crop_w if j < in_w // crop_w - 1 else in_w)

                        padding = 3 // 2
                        pad_i1, pad_i2 = (padding if i1 - padding >= 0 else 0), (padding if i2 + padding <= in_h else 0)
                        pad_j1, pad_j2 = (padding if j1 - padding >= 0 else 0), (padding if j2 + padding <= in_w else 0)

                        crop_feat = feat[:, :, i1 - pad_i1:i2 + pad_i2, j1 - pad_j1:j2 + pad_j2]
                        crop_non_local_feat = self.cs_attn(crop_feat)
                        self.non_local_feat[:, :, i1:i2, j1:j2] = crop_non_local_feat[:, :,
                                                                  pad_i1:crop_non_local_feat.shape[-2] - pad_i2,
                                                                  pad_j1:crop_non_local_feat.shape[-1] - pad_j2]
            else:
                self.non_local_feat = self.cs_attn(feat)

        if self.feat_unfold:
            # 3x3 feature unfolding
            rich_feat = F.unfold(feat, self.unfold_range, padding=self.unfold_range // 2).view(
                bs, in_c * self.unfold_dim, in_h, in_w)
        else:
            rich_feat = feat

        return feat, rich_feat

    def update_scale2mean(self, feat, mod, coord, cell=None):
        """
        Update the Scale2mod table for CMSR testing.

        :param feat: Feature maps (B, C, h, w)
        :param mod: Latent modulations (B, C', h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :return: None
        """

        bs = coord.shape[0]
        # Query rgbs with target scale
        max_pred = self.query_rgb(feat, coord, cell)
        max_pred = max_pred.view(bs * coord.shape[1], -1)

        # Bilinear upsample mod mean to target scale
        mod_mean = torch.mean(torch.abs(mod[:, self.mod_dim // 2:self.mod_dim, :, :]), 1, keepdim=True)
        mod_mean = F.grid_sample(
            mod_mean, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        min_, max_ = 0, 0.5
        samples = [min_ + (max_ - min_) * i / 100 for i in range(101)]
        max_scale = math.sqrt(coord.shape[1] / feat.shape[-2] / feat.shape[-1])
        for scale in self.scale2mean.keys():
            if scale >= max_scale:
                break

            # Query rgbs with current scale
            qh, qw = int(feat.shape[-2] * scale), int(feat.shape[-1] * scale)
            q_coord = make_coord([qh, qw], flatten=False).cuda().view(bs, qh * qw, -1)
            q_cell = torch.ones_like(q_coord)
            q_cell[:, :, 0] *= 2 / qh
            q_cell[:, :, 1] *= 2 / qw
            q_pred = self.query_rgb(feat, q_coord, q_cell)

            # Bilinear upsample rgbs to target scale
            pred = F.grid_sample(
                q_pred.view(bs, qh, qw, -1).permute(0, 3, 1, 2), coord.flip(-1).unsqueeze(1), mode='bilinear',
                padding_mode='border', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            pred = pred.view(bs * coord.shape[1], -1)

            max_sample = self.scale2mean[scale]
            for mid in [i for i in samples]:
                mask_indice = torch.where(torch.abs(mod_mean - mid).flatten() <= 0.001)[0]
                loss = self.loss_fn(pred[mask_indice, :], max_pred[mask_indice, :])

                if loss == loss:
                    if loss <= float(self.mse_threshold):
                        # Fully rendered at current scale
                        samples.remove(mid)
                        max_sample = mid
                    else:
                        break

            # if max_sample < self.scale2mean[scale]:
            #     print(self.scale2mean)
            self.scale2mean[scale] = max_sample if max_sample < self.scale2mean[scale] else self.scale2mean[scale]
            for s in self.scale2mean.keys():
                if s < scale and self.scale2mean[s] > self.scale2mean[scale]:
                    self.scale2mean[s] = self.scale2mean[scale]

        if samples:
            self.scale2mean[max_scale] = samples[-1]
            for s in self.scale2mean.keys():
                if s < max_scale and self.scale2mean[s] > self.scale2mean[max_scale]:
                    self.scale2mean[s] = self.scale2mean[max_scale]

        return self.scale2mean

    def query_rgb_cmsr(self, feat, mod, coord, cell=None):
        """
        Query RGB values of each coordinate using latent modulations and latent codes. (CMSR included)

        :param feat: Feature maps (B, C, h, w)
        :param mod: Latent modulations (B, C', h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :return: Predicted RGBs (B, H * W, 3)
        """

        bs, qn = coord.shape[:2]
        if self.cmsr_log:
            self.total_qn += qn

        mod_mean = torch.mean(torch.abs(mod[:, self.mod_dim // 2:self.mod_dim, :, :]), 1, keepdim=True)

        # Load the Scale2mod table
        scale = math.sqrt(qn / feat.shape[-2] / feat.shape[-1])
        for k, v in self.s2m_tables.items():
            scale2mean = self.s2m_tables[k]
            if k >= scale:
                break

        decode_scales, mask_thresholds = [], [0]
        for s, t in scale2mean.items():
            if s >= scale:
                break
            decode_scales.append(s)
            mask_thresholds.append(t)
        if mask_thresholds[-1] < 1:
            decode_scales.append(scale)
            mask_thresholds.append(1)
        mask_level = len(mask_thresholds) - 1

        i_start, i_end = 0, mask_level - 1
        q_coords, masked_coords, masked_cells, mask_indices = [], [], [], []
        for i in range(mask_level):
            decode_scale = decode_scales[i]
            # Skip decoding if decoding scale < 1
            if decode_scale < 1:
                i_start += 1
                continue

            qh, qw = int(feat.shape[-2] * decode_scale), int(feat.shape[-1] * decode_scale)
            q_coord = F.interpolate(self.feat_coord, size=(qh, qw), mode='bilinear',
                                    align_corners=False, antialias=False).permute(0, 2, 3, 1).view(bs, qh * qw, -1)
            # q_coord = make_coord([qh, qw], flatten=False).cuda().view(bs, qh * qw, -1)

            # Only query coordinates where mod means indicate that they can be decoded to desired accuracy at current scale
            if i == i_end or i == i_start:
                q_mod_mean = F.grid_sample(
                    mod_mean, q_coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                if i == i_end:
                    # Query pixels where mod mean >= min threshold
                    q_mask_indice = torch.where(q_mod_mean.flatten() >= mask_thresholds[i])[0]
                else:
                    # Query pixels where mod mean <= max threshold
                    q_mask_indice = torch.where(q_mod_mean.flatten() <= mask_thresholds[i + 1])[0]
            else:
                # Query pixels where min threshold <= mod mean <= max threshold
                min_, max_ = mask_thresholds[i], mask_thresholds[i + 1]
                mid = (max_ + min_) / 2
                r = (max_ - min_) / 2
                q_mod_mean = F.grid_sample(
                    torch.abs(mod_mean - mid), q_coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_mask_indice = torch.where(q_mod_mean.flatten() <= r)[0]

            mask_indices.append(q_mask_indice)
            if self.cmsr_log:
                self.total_q += len(q_mask_indice) / ((scale / decode_scale) ** 2)
            q_coords.append(q_coord)

            if len(q_mask_indice) <= 0:
                continue

            masked_coords.append(
                q_coord.view(bs * qh * qw, -1)[q_mask_indice, :].view(bs, len(q_mask_indice) // bs, -1))
            masked_cell = torch.ones_like(masked_coords[-1])
            masked_cell[:, :, 0] *= 2 / qh
            masked_cell[:, :, 1] *= 2 / qw
            masked_cell = masked_cell * max(decode_scale / self.max_scale, 1)
            masked_cells.append(masked_cell)

        # CMSR debug log
        if self.cmsr_log:
            print('valid mask: ', self.total_q / self.total_qn)
        pred = self.batched_query_rgb(feat, torch.cat(masked_coords, dim=1), torch.cat(masked_cells, dim=1),
                                      self.query_bsize)
        pred += F.grid_sample(self.inp, torch.cat(masked_coords, dim=1).flip(-1).unsqueeze(1), mode='bilinear', \
                              padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        # Merge rgb predictions at different scales
        ret = self.inp
        skip_indice_i = 0
        for i in range(i_start, mask_level):
            decode_scale = decode_scales[i]
            qh, qw = int(feat.shape[-2] * decode_scale), int(feat.shape[-1] * decode_scale)

            q_mask_indice = mask_indices[i - i_start]
            q_coord = q_coords[i - i_start]
            if len(q_mask_indice) <= 0:
                continue

            # Bilinear upsample predictions at last scale
            ret = F.grid_sample(
                ret, q_coord.flip(-1).unsqueeze(1), mode='bilinear',
                padding_mode='border', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1).contiguous().view(bs * qh * qw, -1)

            # Merge predictions at current scale
            ret[q_mask_indice, :] = pred[:, skip_indice_i:skip_indice_i + len(q_mask_indice) // bs, :].view(
                len(q_mask_indice), -1)
            skip_indice_i += len(q_mask_indice)

            if i < mask_level - 1:
                ret = ret.view(bs, qh, qw, -1).permute(0, 3, 1, 2)
            else:
                if decode_scales[-1] < scale and qh * qw != qn:
                    ret = F.grid_sample(
                        ret.view(bs, qh, qw, -1).permute(0, 3, 1, 2), coord.flip(-1).unsqueeze(1), mode='bilinear',
                        padding_mode='border', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                ret = ret.view(bs, qn, -1)
        return ret

    def query_latent(self, feat, scale=None):
        """

        :param feat: Feature maps (B, C, h, w)
        :param scale: Cell areas (B, H * W, 2)
        :return:
        """

        # TODO: batched_query_latent() to avoid OOM

        bs, c, h, w = feat.shape
        base_c = 64
        ur, er = self.unfold_range, 2
        feat_coord = self.feat_coord

        # Field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # 2x2 feature ensemble
        if self.local_size == 1:
            v_lst = [(0, 0)]
        else:
            v_lst = [(i, j) for i in range(-1, 2, 4 - self.local_size) for j in range(-1, 2, 4 - self.local_size)]

        preds_k, preds_v = [], []
        for v in v_lst:
            vx, vy = v[0], v[1]

            # Key and value
            t, l = round(vy / 2 + 0.5), round(vx / 2 + 0.5)
            key = value = feat.view(bs, ur, ur, base_c, h, w)[:, t:t + er, l:l + er, :, :, :].contiguous()
            key = key.view(bs, er * er * base_c, h * w).permute(0, 2, 1)
            value = value.view(bs, er * er * base_c, h, w)
            if self.non_local_attn:
                value = torch.cat([value, self.non_local_feat], dim=1)
                value = value.view(bs, er * er * base_c + base_c, h * w).permute(0, 2, 1)

            coord_q = feat_coord.view(bs, feat_coord.shape[1], h * w).permute(0, 2, 1)
            coord_k = coord_q.clone()
            coord_k[:, :, 0] += vx * rx / feat.shape[-2]  # + eps_shift
            coord_k[:, :, 1] += vy * ry / feat.shape[-1]  # + eps_shift

            bs, q = coord_q.shape[:2]
            Q, K = coord_q, coord_k
            rel = Q - K
            rel[:, :, 0] *= feat.shape[-2]  # without mul
            rel[:, :, 1] *= feat.shape[-1]
            inp = rel

            scale_ = scale[:, :feat.shape[-2] * feat.shape[-1], :].clone()
            scale_[:, :, 0] *= feat.shape[-2]
            scale_[:, :, 1] *= feat.shape[-1]

            weight_k = self.imnet_k(torch.cat([key, inp, scale_], dim=-1).view(bs * q, -1)).view(bs, q, -1)
            weight_v = self.imnet_v(torch.cat([value, inp, scale_], dim=-1).view(bs * q, -1)).view(bs, q, -1)

            preds_k.append((key * weight_k).view(bs, q, -1))
            preds_v.append((value * weight_v).view(bs, q, -1))

        preds_k = torch.stack(preds_k, dim=-1)
        preds_v = torch.stack(preds_v, dim=-2)

        query = feat.view(bs, c, h * w).permute(0, 2, 1).contiguous().view(bs * h * w, -1)
        query = self.prenet_q(query).view(bs, h * w, -1).unsqueeze(2)

        # Query modulations
        attn = (query @ preds_k)
        inp_q = ((attn / self.softmax_scale).softmax(dim=-1) @ preds_v)
        mod = self.hypernet_q(inp_q.view(bs * q, -1)).view(bs, h, w, -1).permute(0, 3, 1, 2)
        if self.mod_input:
            self.inp_q = mod[:, self.mod_dim:, :, :]
        else:
            self.inp_q = inp_q.view(bs, h, w, -1).permute(0, 3, 1, 2)

        return mod, preds_k, preds_v

    def query_rgb(self, feat, coord, scale=None):
        """
        Query RGB values of each coordinate using latent modulations and latent codes. (without CMSR)

        :param feat: Feature maps (B, C, h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param scale: Cell areas (B, H * W, 2)
        :return: Predicted RGBs (B, H * W, 3)
        """

        bs, q = coord.shape[:2]
        feat_coord = self.feat_coord

        local_size = self.local_size
        if local_size == 1:
            v_lst = [(0, 0)]
        else:
            v_lst = [(i, j) for i in range(-1, 2, 4 - local_size) for j in range(-1, 2, 4 - local_size)]

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        eps_shift = 1e-6

        preds = []
        coords = []
        areas = []
        for v in v_lst:
            vx, vy = v[0], v[1]

            coord_ = coord.clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

            q_coord = F.grid_sample(
                feat_coord, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]

            inp = F.grid_sample(
                feat, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1).contiguous()
            inp = torch.cat([inp, rel_coord], dim=-1).view(bs * q, -1)

            # Use latent modulations to boost the render mlp
            if self.training:
                q_mod = F.grid_sample(
                    self.mod, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1).contiguous().view(bs * q, -1)

                pred = self.imnet_q(inp, mod=q_mod).view(bs, q, -1)  # [16, 2304, 3]
                preds.append(pred)
            else:
                pred0 = self.imnet_q(inp, only_layer0=True)
                preds.append(pred0)
                coords.append(coord_)

            area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
            areas.append(area + 1e-9)

        if not self.training:
            # Upsample modulations of each layer seperately, avoiding OOM
            preds = self.imnet_q(torch.cat(preds, dim=0), mod=self.mod, coord=torch.cat(coords, dim=1),
                                 skip_layer0=True).view(local_size ** 2, bs, q, -1)

        tot_area = torch.stack(areas).sum(dim=0)
        if local_size == 2:
            t = areas[0];areas[0] = areas[3];areas[3] = t
            t = areas[1];areas[1] = areas[2];areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

    def query_rgb_fast(self, feat, coord, cell=None):
        """
        Query RGB values of each coordinate using latent modulations and latent codes. (without CMSR)

        :param feat: Feature maps (B, C, h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :return: Predicted RGBs (B, H * W, 3)
        """

        if self.local_size == 2:
            vx_lst = [-1, 1]  # left, right
            vy_lst = [-1, 1]  # top, bottom
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # Field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # Prepare coordinates and cells
        bs, q = coord.shape[:2]
        ls = len(vx_lst) * len(vy_lst)
        h, w = feat.shape[-2:]
        coords, rel_coords, rel_cells, areas = [], [], [], []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                coords.append(coord_)

                q_coords = F.grid_sample(
                    self.feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coords
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w
                rel_coords.append(rel_coord)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        coords = torch.cat(coords, dim=1)
        rel_coords = torch.cat(rel_coords, dim=1)

        # Upsample lr feat to hr feat
        inp = F.grid_sample(
            feat, coords.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        inp = torch.cat([inp, rel_coords], dim=-1)
        inp = inp.view(bs * ls * q, -1)

        # Upsample modulations of each layer seperately, avoiding OOM
        preds = self.imnet_q(inp, mod=self.mod, coord=coords).view(bs, ls, q, -1).permute(1, 0, 2, 3)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_size == 2:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

    def batched_query_rgb(self, feat, coord, cell, bsize):
        """Batched predict.

        Args:
            feat (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """

        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb_fast(feat, coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, inp, coord=None, cell=None, inp_coord=None, bsize=None):
        """
        Forward function.

        :param inp: Input image (B, h * w, 3)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :param inp_coord: Input coordinates (B, h * w, 2)
        :param bsize: Number of pixels in each query
        :return: Predicted image (B, H * W, 3)
        """

        self.inp = inp
        if coord is None and cell is None:
            # Evaluate the efficiency of encoder only
            feat = self.encoder(inp)
            return None

        # Adjust the number of query pixels for different GPU memory limits
        # Using lmf, we can query a 2k image simultaneously with 6GB GPU memory
        self.query_bsize = bsize if bsize is not None else int(1280 * 720)
        self.query_bsize = math.ceil(coord.shape[1] / math.ceil(coord.shape[1] / self.query_bsize))

        feat, rich_feat = self.gen_feats(inp, inp_coord)
        self.mod, self.preds_k, self.preds_v = self.query_latent(rich_feat, cell)

        if self.training:
            pred = self.query_rgb(self.inp_q, coord, cell)
        else:
            if self.cmsr and self.updating_cmsr:
                # Update the Scale2mod Table for CMSR
                self.update_scale2mean(self.inp_q, self.mod, coord, cell)
                return None

            out_of_distribution = coord.shape[1] > (self.max_scale ** 2) * inp.shape[-2] * inp.shape[-1]
            if self.cmsr and out_of_distribution:
                # Only use CMSR for out-of-training scales
                pred = self.query_rgb_cmsr(self.inp_q, self.mod, coord, cell)
            else:
                pred = self.batched_query_rgb(self.inp_q, coord, cell, self.query_bsize)
                pred += F.grid_sample(inp, coord.flip(-1).unsqueeze(1), mode='bilinear', \
                                      padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

            # self.t_total.append(time.time() - self.t1)
            #if len(self.t_total) >= 100:
            #    print(sum(self.t_total[1:]) / (len(self.t_total) - 1))

        return pred
