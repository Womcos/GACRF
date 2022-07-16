from .base import *
from .fcn import *
from .psp import *
from .fcfpn import *
from .deeplab import *
from .upernet import *
from .danet import *
from .deeplab_GACRF import *

def get_segmentation_model(name, **kwargs):
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'fcfpn': get_fcfpn,
        'upernet': get_upernet,
        'danet': get_danet,
        'deeplab': get_deeplab,
        'gacrf': get_deeplabv3_GACRF,
    }
    return models[name.lower()](**kwargs)
