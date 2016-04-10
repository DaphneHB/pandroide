# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:38:36 2016

@author: daphnehb
"""

import cv2
import os
import numpy as np

IEME_TAG = 0
# times list for different algorithms used to filter img
TPS_NOUS = []
TPS_ELIAS = []
TPS_EMILIAS = []

# to check result classification
TRUE_POS = []
TRUE_NEG = []
FALSE_POS = []
FALSE_NEG = []

# constants
ITERATIONS = 50
SIZE_X = 804  # 2592
SIZE_Y = 603  # 1944
ABS_PATH_PRINC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_PATH = ABS_PATH_PRINC + "/data/img/"
PLOT_PATH = ABS_PATH_PRINC + "/data/plots/"
FILE_PATH = ABS_PATH_PRINC + "/data/files/"

HIERARCHY_TREE = cv2.RETR_CCOMP

# in pixel
SUPP_MARGIN = 10
# in cm
OCCLUSION_MARGIN = 1
TAG_HEIGHT = 10.5
TAG_WIDTH = 7.8

# different convolution matrix
k_contraste = np.array([[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, 5, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]])
k_bords = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, -4, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
kernel_float32 = np.ones((5, 5), np.float32) / 25
kernel_uint8 = np.ones((5, 5), np.uint8)

# a tag protype
tag_alone = cv2.imread('tag.png', 0)

