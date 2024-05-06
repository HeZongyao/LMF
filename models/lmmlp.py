import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


@register('lmmlp')
class LMMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_depth,
                 mod_scale=True, mod_shift=True, mod_up_merge=False, use_conv=False):
        super().__init__()
        self.hidden_depth = hidden_depth
        self.hidden_dim = hidden_dim

        # Modulation configs
        self.mod_scale = mod_scale
        self.mod_shift = mod_shift
        self.mod_dim = 0
        # If we modulate both scale and shift, we have twice the number of modulations at every layer and feature
        self.mod_dim += hidden_dim if self.mod_scale else 0
        self.mod_dim += hidden_dim if self.mod_shift else 0

        # For faster inference, set to True if upsample scale mod and shift mod together.
        self.mod_up_merge = mod_up_merge and self.mod_scale and self.mod_shift

        layers = []
        lastv = in_dim
        for _ in range(hidden_depth):
            if use_conv:
                layers.append(nn.Conv2d(lastv, hidden_dim, 1))
            else:
                layers.append(nn.Linear(lastv, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            lastv = hidden_dim
        if use_conv:
            layers.append(nn.Conv2d(lastv, out_dim, 1))
        else:
            layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, mod=None, coord=None, only_layer0=False, skip_layer0=False):
        shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])

        if only_layer0:
            return self.layers[0](x)

        if coord is None:
            mod = mod.view(-1, mod.shape[-1])

        # Split modulations into shifts and scales and apply them to hidden features
        mid_dim = (
            self.mod_dim * self.hidden_depth // 2 if (self.mod_scale and self.mod_shift) else 0
        )

        for idx, module in enumerate(self.layers):
            if not (skip_layer0 and idx == 0):
                x = module(x)

            if idx == self.hidden_depth * 2 or idx % 2 == 1:
                # skip output linear layer or hidden activation layer
                continue

            start, end = (idx // 2) * self.hidden_dim, ((idx // 2) + 1) * self.hidden_dim

            # Use modulations on hidden linear layer outputs
            if self.mod_up_merge and coord is not None:
                # Upsample scale mod and shift mod together when GPU memory is sufficient.
                bs, q = coord.shape[:2]
                q_mod = F.grid_sample(
                    torch.cat([mod[:, start: end, :, :], mod[:, mid_dim + start: mid_dim + end, :, :]], dim=1),
                    coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous().view(bs * q, -1)
                x *= q_mod[:, :self.hidden_dim]
                x += q_mod[:, self.hidden_dim:]
            else:
                if self.mod_scale:
                    # Shape (b * h * w, hidden_dim). Note that we add 1 so modulations remain zero centered
                    if coord is not None:
                        bs, q = coord.shape[:2]
                        x *= (F.grid_sample(
                            mod[:, start: end, :, :], coord.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=False)[:, :, 0, :]
                              .permute(0, 2, 1).contiguous().view(bs * q, -1) + 1.0)
                    else:
                        x *= (mod[:, start: end] + 1.0)

                if self.mod_shift:
                    # Shape (b * h * w, hidden_dim)
                    if coord is not None:
                        bs, q = coord.shape[:2]
                        x += F.grid_sample(
                            mod[:, mid_dim + start: mid_dim + end, :, :], coord.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=False)[:, :, 0, :] \
                            .permute(0, 2, 1).contiguous().view(bs * q, -1)
                    else:
                        x += mod[:, mid_dim + start: mid_dim + end]

            # Broadcast scale and shift across x
            # scale, shift = 1.0, 0.0
            #x = scale * x + shift

        return x.view(*shape, -1)