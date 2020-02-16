"""Adjust configuration here for settings, to be loaded in other code.

This is used for surgical robotics and any Python2 code. DO NOT MERGE WITH
load_config in the neural net code. So there are two config files.

Do not use the camera code here because now the camera uses python3.
"""
import os
import cv2
import sys
import time
import pickle
import numpy as np
from os.path import join
import utils as U

# Colors for cv2.
BLUE  = (255,0,0)
GREEN = (0,255,0)
RED   = (0,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)

# ---------------------------------------------------------------------------- #
# WHERE DVRK CODE SAVES IMAGES -- must be same as in: image_manip/load_config.py
# ---------------------------------------------------------------------------- #
DVRK_IMG_PATH = 'dir_for_imgs/'

# ---------------------------------------------------------------------------- #
# CALIBRATION FILE
# ---------------------------------------------------------------------------- #

# For RSS 2020 submission.
#CALIB_FILE = 'tests/mapping_2020_01_16_psm1'

# For IROS 2020 (re)submission.
CALIB_FILE = 'tests/mapping_2020_02_16_psm1'

ROW_BOARD = 6
COL_BOARD = 6

# In meters, also due to checkerboard height! We have a foam rubber, it's OK. :-)
# Decrease this value to decrease the height the gripper lands at.
CLOTH_HEIGHT = -0.008

DATA_SQUARE = U.load_mapping_table(row_board=ROW_BOARD,
                                   column_board=COL_BOARD,
                                   file_name=CALIB_FILE,
                                   cloth_height=CLOTH_HEIGHT)
