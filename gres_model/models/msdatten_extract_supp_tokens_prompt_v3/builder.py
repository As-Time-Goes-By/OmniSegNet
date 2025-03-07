import torch
import torch.nn as nn
# from .models import *
# from .backbones import *
# from .msdeform_decoder import VLMSDeformAttnPixelDecoder
# from .msdeform_decoder_dit import VLMSDeformAttnPixelDecoder
# from .msdeform_decoder_vrp import VLMSDeformAttnPixelDecoder
from .GRES import GRES
def _segm_gres(cfg):
    # initialize the SwinTransformer backbone with the specified version

    model_components = GRES.from_config(cfg)
    model = GRES(**model_components)
    return model

def gres(cfg):
    return _segm_gres(cfg)