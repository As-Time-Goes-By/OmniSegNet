import torch
import torch.nn as nn

from .OmniSegNet import OmniSegNet
def _segm_OmniSegNet(cfg):
    # initialize the SwinTransformer backbone with the specified version

    model_components = OmniSegNet.from_config(cfg)
    model = OmniSegNet(**model_components)
    return model

def omnisegnet(cfg):
    return _segm_OmniSegNet(cfg)