# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:38:36 2016

@author: daphnehb
"""

import cv2
import os
import numpy as np


# liste des temps
TPS_NOUS = []
TPS_ELIAS = []
TPS_EMILIAS = []

ITERATIONS = 50

# constantes
SIZE_X = 804  # 2592
SIZE_Y = 603  # 1944
ABS_PATH_PRINC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_PATH = ABS_PATH_PRINC + "/data/img/"
PLOT_PATH = ABS_PATH_PRINC + "/data/plots/"
FILE_PATH = ABS_PATH_PRINC + "/data/files/"

HIERARCHY_TREE = cv2.RETR_CCOMP

# differentes matrices de convolution
k_contraste = np.array([[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, 5, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]])
k_bords = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, -4, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
kernel_float32 = np.ones((5, 5), np.float32) / 25
kernel_uint8 = np.ones((5, 5), np.uint8)

tag_seul = cv2.imread('tag.png', 0)

