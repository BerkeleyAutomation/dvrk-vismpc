"""Adjust configuration here for settings.

This is used in the neural network loading code. THIS WILL REQUIRE PYTHON 3, so
DO NOT MERGE WITH OUR OTHER CONFIG FILE in the top-level repository.
"""
import os
import sys
import time
import pickle
import numpy as np
from os.path import join

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)

# ---------------------------------------------------------------------------- #
# WHERE THE DVRK CODE SAVES IMAGES
# ---------------------------------------------------------------------------- #
DVRK_IMG_PATH = 'dir_for_imgs/'

# ---------------------------------------------------------------------------- #
# ADJUST WHICH NEURAL NETWORK WE WANT TO USE
# And be sure we adjust if we use color or depth for `run.py`.
# Update for Jan 2020+, we just have one network now.
# ---------------------------------------------------------------------------- #

# September 2019 experiments.
HEAD   = '/home/davinci0/seita/dvrk_python/nets/'
POL_01 = 'openai-2019-09-02-23-03-41-443793_tier1_color_50k/checkpoints/00249'
POL_02 = 'openai-2019-09-02-23-01-55-146225_tier1_depth_50k/checkpoints/00249'
POL_03 = 'openai-2019-09-01-20-37-45-609860_tier2_color_50k/checkpoints/00249'
POL_04 = 'openai-2019-09-04-12-56-54-072948_tier2_depth_50k/checkpoints/00249'
POL_05 = 'openai-2019-09-02-19-30-13-323241_tier3_color_50k/checkpoints/00249'
POL_06 = 'openai-2019-09-02-19-25-40-802588_tier3_depth_50k/checkpoints/00249'

# RGBD baselines.
HEAD   = '/home/davinci/seita/policies-cloth-sim-iros2020/'
POL_07 = 'openai-2020-02-11-15-52-56-391653_tier1_rgbd_50k/checkpoints/00249'
POL_08 = 'openai-2020-02-12-14-11-03-092908_tier2_rgbd_50k/checkpoints/00249'
POL_09 = 'openai-2020-02-15-19-37-59-883528_tier3_rgbd_50k/checkpoints/00249'

WHICH_POLICY = POL_09  # ADJUST!!
NET_FILE = join(HEAD, WHICH_POLICY)

# RSS 2020.
#HEAD2 = '/home/davinci/seita/policies-cloth-sim-rss2020'
#WHICH_POLICY = 'openai-2020-01-15-11-38-10-349814/checkpoints/00549'
#NET_FILE = join(HEAD2, WHICH_POLICY)

# ---------------------------------------------------------------------------- #
# ADJUST WHICH PATH WE WANT TO USE FOR TESTING
# ---------------------------------------------------------------------------- #
# This is only if we're testing some known images. Comment out if not.
#IMG_HEAD = '/home/davinci0/images'
#image_files = sorted(
#    [join(IMG_HEAD,x) for x in os.listdir(IMG_HEAD) \
#        if 'resized' in x and '.png' in x]
#)
#TEST_IMAGE_FILES = image_files[:20]
