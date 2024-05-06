import argparse
import os
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
from utils import make_coord


def build_scale2mean(loader, model, scale=4, data_norm=None, window_size=0, trained_scale=4):
    """


    :param loader:
    :param model:
    :param scale:
    :param data_norm:
    :param window_size:
    :param trained_scale:
    :return:
    """

    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div

        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]

            coord = make_coord((scale * (h_old + h_pad), scale * (w_old + w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            coord = batch['coord']
            cell = batch['cell']

        cell *= max(scale / trained_scale, 1)
        with torch.no_grad():
            with torch.no_grad():
                model(inp=inp, coord=coord, cell=cell)

    return model.scale2mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--window', default='0')
    parser.add_argument('--trained_scale', default='4')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--mse_threshold', default=0.00002)
    parser.add_argument('--save_path')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8, pin_memory=True)

    scale = int(config.get('eval_type').split('-')[1])
    try:
        with open(args.save_path, 'r') as f:
            s2m_tables = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        s2m_tables = {}
    if scale not in s2m_tables:
        s2m_tables[scale] = {s: 1 for s in [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32] if s <= scale}
    s2m_tables = {k: s2m_tables[k] for k in sorted(s2m_tables)}
    cmsr_spec = {
        "mse_threshold": float(args.mse_threshold),
        "path": args.save_path,
        "s2m_tables": s2m_tables,
        "updating_scale": scale,
    }

    model_spec = torch.load(args.model)['model']
    model_spec["args"]["cmsr_spec"] = cmsr_spec
    model = models.make(model_spec, load_sd=True).cuda()

    scale2mean = build_scale2mean(loader,
                                  model,
                                  scale=scale,
                                  data_norm=config.get('data_norm'),
                                  window_size=int(args.window))

    s2m_tables[scale] = scale2mean
    with open(args.save_path, 'w') as file:
        yaml.dump(s2m_tables, file)
    print(f'Scale2Mod Table with scale {scale} and mse {args.mse_threshold}: ', scale2mean)
