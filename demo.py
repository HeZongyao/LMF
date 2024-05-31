import argparse
import os
from PIL import Image

import yaml
import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--fast', default=False)  # Set fast to True for LMF
    parser.add_argument('--cmsr', default=False)
    parser.add_argument('--cmsr_mse', default=0.00002)
    parser.add_argument('--cmsr_path')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Maximum scale factor during training
    scale_max = 4

    if args.cmsr:
        try:
            # Test with CMSR
            with open(args.cmsr_path, 'r') as f:
                s2m_tables = yaml.load(f, Loader=yaml.FullLoader)
            cmsr_spec = {
                "mse_threshold": float(args.cmsr_mse),
                "path": args.cmsr_path,
                "s2m_tables": s2m_tables,
                "log": False,
            }
        except FileNotFoundError:
            cmsr_spec = None
    else:
        cmsr_spec = None

    model_spec = torch.load(args.model)['model']
    model_spec["args"]["cmsr_spec"] = cmsr_spec
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()
    
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    h = int(img.shape[-2] * int(args.scale))
    w = int(img.shape[-1] * int(args.scale))
    scale = h / img.shape[-2]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    cell_factor = max(scale/scale_max, 1)
    if args.fast:
        with torch.no_grad():
            pred = model(((img - 0.5) / 0.5).cuda().unsqueeze(0),
                         coord.unsqueeze(0), cell_factor * cell.unsqueeze(0))[0]
    else:
        pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
                               coord.unsqueeze(0), cell_factor * cell.unsqueeze(0), bsize=30000)[0]

    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
