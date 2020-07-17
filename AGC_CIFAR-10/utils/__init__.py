from .parsing import parse_args
from .misc import get_logger, Option
from .zca_bn import ZCA

import tensorflow as tf
import os, sys

def init():
    gpu_thread_count = 2
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    print('PY', sys.version)
    print('TF', tf.__version__)