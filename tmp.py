import torch
import numpy as np
import os
from config.config import Config
from models.cyclegan import create_model
from models.cyclegan import util
from models.

cfg = Config()

def get_cgan(cfg):
    m = create_model(cfg.cyclegan_cfg)
    m.setup(cfg.cyclegan_cfg)
    m.eval()

def test_cgan(cgan):
    dummy_d = torch.rand([800, 800, 3])  # rgb from nerf
    dummy_d = dummy_d.permute([2, 0, 1])

    cgan.set_input_from_nerf(dummy_d)
    img_path = 'results/cyclegan/'
    cgan.test()
    visuals = m.get_current_visuals()

    for label, im_data in visuals.items():
        im = im_data.data
        im = im.cpu().float().numpy()  # convert it into a numpy array
        im = (np.transpose(im, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        im = im.astype(np.uint8)
        save_path = os.path.join(img_path, 'test0.png')
        util.save_image(im, save_path, aspect_ratio=1.0)

def get_nerf(cfg:Config):
    if cfg.nerf == "DNerfParticle":
        nerf = DNeRFParticle(cfg).to(cfg.device)
    else:
        raise ValueError(f"Unsupported Nerf {cfg.nerf}")
    return nerf
cgan = get_cgan(cfg)
print("it works")