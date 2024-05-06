import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('lmliif')
class LMLIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hypernet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True,
                 mod_input=False, cmsr_spec=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.unfold_range = 3 if feat_unfold else 1
        self.unfold_dim = self.unfold_range * self.unfold_range
        self.cell_decode = cell_decode
        self.max_scale = 4  # Max training scale
        self.mod_input = mod_input  # Set to True if use compressed latent code

        self.encoder = models.make(encoder_spec)

        # Use latent MLPs to generate modulations for the render MLP
        if hypernet_spec is not None:
            hypernet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                hypernet_in_dim *= self.unfold_dim  # + (1 if self.feat_reserve else 0)
            if self.cell_decode:
                hypernet_in_dim += 2

            self.mod_dim = 0
            self.mod_dim += imnet_spec['args']['hidden_dim'] if imnet_spec['args']['mod_scale'] else 0
            self.mod_dim += imnet_spec['args']['hidden_dim'] if imnet_spec['args']['mod_shift'] else 0
            self.mod_dim *= imnet_spec['args']['hidden_depth']
            hypernet_out_dim = self.mod_dim
            if self.mod_input:
                hypernet_out_dim += imnet_spec['args']['hidden_dim']
            self.hypernet = models.make(hypernet_spec, args={'in_dim': hypernet_in_dim, 'out_dim': hypernet_out_dim})
        else:
            self.hypernet = None

        # Render MLP
        if imnet_spec is not None:
            imnet_in_dim = imnet_spec['args']['hidden_dim'] if self.mod_input else self.encoder.out_dim
            #if self.feat_unfold and not self.use_modulations:
            #    imnet_in_dim *= self.unfold_dim

            imnet_in_dim += 2  # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim, 'mod_up_merge': False})
        else:
            self.imnet = None

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
        :return: Feature maps (B, C, h, w)
        """

        feat = self.encoder(inp)

        if inp_coord is not None:
            self.feat_coord = inp_coord.permute(0, 3, 1, 2)
        elif self.training or self.feat_coord is None or self.feat_coord.shape[-2] != feat.shape[-2] \
                or self.feat_coord.shape[-1] != feat.shape[-1]:
            self.feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
                .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        if self.feat_unfold:
            # 3x3 unfolding
            rich_feat = F.unfold(feat, self.unfold_range, padding=self.unfold_range // 2).view(
                feat.shape[0], feat.shape[1] * self.unfold_dim, feat.shape[2], feat.shape[3])
        else:
            rich_feat = feat

        return feat, rich_feat

    def gen_modulations(self, feat, cell=None):
        """
        Generate latent modulations using the latent MLP.

        :param feat: Feature maps (B, C, h, w)
        :param cell: Cell areas (B, H * W, 2)
        :return: Latent modulations (B, C', h, w)
        """

        bs, c, h, w = feat.shape

        initial_mod = feat.permute(0, 2, 3, 1).contiguous().view(bs, -1, c)
        if self.cell_decode:
            # use relative height, width info
            rel_cell = cell.clone()[:, :h * w, :]
            rel_cell[:, :, 0] *= h
            rel_cell[:, :, 1] *= w
            initial_mod = torch.cat([initial_mod, rel_cell], dim=-1)

        mod = self.hypernet(initial_mod)
        mod = mod.view(feat.shape[0], feat.shape[-2], feat.shape[-1], -1).permute(0, 3, 1, 2).contiguous()

        return mod

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
        max_pred = self.query_rgb(feat, mod, coord, cell)
        max_pred = max_pred.view(bs * coord.shape[1], -1)

        # Bilinear upsample mod mean to target scale
        mod_mean = torch.mean(torch.abs(mod[:, self.mod_dim // 2:self.mod_dim, :, :]), 1, keepdim=True)
        mod_mean = F.grid_sample(
            mod_mean, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        min_, max_ = 0, 1
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
            q_pred = self.query_rgb(feat, mod, q_coord, q_cell)

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
                # print('mean:', mid, '| indices:', len(mask_indice) / coord.shape[-1], '| loss:', loss.item())

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

            masked_coords.append(q_coord.view(bs * qh * qw, -1)[q_mask_indice, :].view(bs, len(q_mask_indice) // bs, -1))
            masked_cell = torch.ones_like(masked_coords[-1])
            masked_cell[:, :, 0] *= 2 / qh
            masked_cell[:, :, 1] *= 2 / qw
            masked_cell = masked_cell * max(decode_scale / self.max_scale, 1)
            masked_cells.append(masked_cell)

        # CMSR debug log
        if self.cmsr_log:
            print('Valid mask: ', self.total_q / self.total_qn)
        pred = self.batched_query_rgb(feat, mod, torch.cat(masked_coords, dim=1),
                                      torch.cat(masked_cells, dim=1), self.query_bsize)
        #pred = self.query_rgb(feat, mod, torch.cat(masked_coords, dim=1), torch.cat(masked_cells, dim=1))

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
            ret[q_mask_indice, :] = pred[:, skip_indice_i:skip_indice_i + len(q_mask_indice) // bs, :].view(len(q_mask_indice), -1)
            skip_indice_i += len(q_mask_indice)

            if i < mask_level - 1:
                ret = ret.view(bs, qh, qw, -1).permute(0, 3, 1, 2)
            else:
                if decode_scales[-1] < scale:
                    ret = F.grid_sample(
                        ret.view(bs, qh, qw, -1).permute(0, 3, 1, 2), coord.flip(-1).unsqueeze(1), mode='bilinear',
                        padding_mode='border', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                ret = ret.view(bs, qn, -1)

        return ret

    def query_rgb(self, feat, mod, coord, cell=None):
        """
        Query RGB values of each coordinate using latent modulations and latent codes. (without CMSR)

        :param feat: Feature maps (B, C, h, w)
        :param mod: Latent modulations (B, C', h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :return: Predicted RGBs (B, H * W, 3)
        """

        if self.imnet is None:
            return F.grid_sample(
            self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
            padding_mode='border', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1).contiguous()

        local_ensemble = self.local_ensemble
        if local_ensemble:
            vx_lst = [-1, 1]  # left, right
            vy_lst = [-1, 1]  # top, bottom
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        bs, q = coord.shape[:2]

        if not self.training:
            coords = []
            for vx in vx_lst:
                for vy in vy_lst:
                    coord_ = coord.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coords.append(coord_)
            coords = torch.cat(coords, dim=1)
            coords.clamp_(-1 + 1e-6, 1 - 1e-6)

            q_coords = F.grid_sample(
                self.feat_coord, coords.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)

        idx = 0
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                if not self.training:
                    coord_ = coords[:, idx * coord.shape[1]:(idx + 1) * coord.shape[1], :]
                    rel_coord = coord - q_coords[:, idx * coord.shape[1]:(idx + 1) * coord.shape[1], :]
                    rel_coord[:, :, 0] *= feat.shape[-2]
                    rel_coord[:, :, 1] *= feat.shape[-1]

                    idx += 1
                else:
                    coord_ = coord.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                    q_coord = F.grid_sample(
                        self.feat_coord, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                    rel_coord = coord - q_coord
                    rel_coord[:, :, 0] *= feat.shape[-2]
                    rel_coord[:, :, 1] *= feat.shape[-1]

                # q_feat
                inp = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                inp = torch.cat([inp, rel_coord], dim=-1)
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)
                inp = inp.view(bs * q, -1)

                # Use latent modulations to boost the render mlp
                if self.training:
                    q_mod = F.grid_sample(
                        mod, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1).contiguous().view(bs * q, -1)
                    pred = self.imnet(inp, mod=q_mod).view(bs, q, -1)
                    preds.append(pred)
                else:
                    pred0 = self.imnet(inp, only_layer0=True)
                    preds.append(pred0)
                    # coords.append(coord_)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        if not self.training:
            # Upsample modulations of each layer seperately, avoiding OOM
            preds = self.imnet(torch.cat(preds, dim=0), mod=mod, coord=coords,
                               skip_layer0=True).view(len(vx_lst) * len(vy_lst), bs, q, -1)

        tot_area = torch.stack(areas).sum(dim=0)
        if local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

    def query_rgb_fast(self, feat, mod, coord, cell=None):
        """
        Query RGB values of each coordinate using latent modulations and latent codes. (without CMSR)

        :param feat: Feature maps (B, C, h, w)
        :param mod: Latent modulations (B, C', h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :return: Predicted RGBs (B, H * W, 3)
        """

        if self.imnet is None:
            return F.grid_sample(
                self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
                padding_mode='border', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1).contiguous()

        if self.local_ensemble:
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

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= h
                    rel_cell[:, :, 1] *= w
                    rel_cells.append(rel_cell)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        coords = torch.cat(coords, dim=1)
        rel_coords = torch.cat(rel_coords, dim=1)
        if self.cell_decode:
            rel_cells = torch.cat(rel_cells, dim=1)

        # Upsample lr feat to hr feat
        inp = F.grid_sample(
            feat, coords.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        inp = torch.cat([inp, rel_coords], dim=-1)
        if self.cell_decode:
            inp = torch.cat([inp, rel_cells], dim=-1)
        inp = inp.view(bs * ls * q, -1)

        # Upsample modulations of each layer seperately, avoiding OOM
        preds = self.imnet(inp, mod=mod, coord=coords).view(bs, ls, q, -1).permute(1, 0, 2, 3)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

    def batched_query_rgb(self, feat, mod, coord, cell, bsize):
        """
        Query RGB values of each coordinate batch using latent modulations and latent codes.

        :param feat: Feature maps (B, C, h, w)
        :param mod: Latent modulations (B, C', h, w)
        :param coord: Coordinates (B, H * W, 2)
        :param cell: Cell areas (B, H * W, 2)
        :param bsize: Number of pixels in each query
        :return: Predicted RGBs (B, H * W, 3)
        """

        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb_fast(feat, mod, coord[:, ql: qr, :], cell[:, ql: qr, :])
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
        # Using lmf, we can query a 4k image simultaneously with 12GB GPU memory
        self.query_bsize = bsize if bsize is not None else int(2160 * 3840 * 0.5)
        self.query_bsize = math.ceil(coord.shape[1] / math.ceil(coord.shape[1] / self.query_bsize))

        feat, rich_feat = self.gen_feats(inp, inp_coord)
        # t1 = time.time()

        mod = self.gen_modulations(rich_feat, cell)
        if self.mod_input:
            feat = mod[:, self.mod_dim:, :, :]

        if self.training:
            out = self.query_rgb(feat, mod, coord, cell)
        else:
            if self.cmsr and self.updating_cmsr:
                # Update the Scale2mod Table for CMSR
                self.update_scale2mean(feat, mod, coord, cell)
                return None

            out_of_distribution = coord.shape[1] > (self.max_scale ** 2) * inp.shape[-2] * inp.shape[-1]
            if self.cmsr and out_of_distribution:
                # Only use CMSR for out-of-training scales
                out = self.query_rgb_cmsr(feat, mod, coord, cell)
            else:
                out = self.batched_query_rgb(feat, mod, coord, cell, self.query_bsize)

            # self.t_total.append(time.time() - t1)
            #if len(self.t_total) >= 100:
            #    print(sum(self.t_total[1:]) / (len(self.t_total) - 1))

        return out
