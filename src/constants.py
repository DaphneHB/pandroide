# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:38:36 2016

@author: daphnehb
"""
from default_cst import *
import os
import numpy as np

IEME_TAG = 0

# times list for different algorithms used to filter img
TPS_NOUS = []
TPS_ELIAS = []
TPS_EMILIAS = []

# times list for different algorithms used to filter img
TPS_OTSU = []
TPS_FIX = []
TPS_CALC = []

# to check result classification
TRUE_POS = 0
TRUE_NEG = 0
FALSE_POS = 0
FALSE_NEG = 0

#to test different parameters of Canny's algo
CANNY_VIDEO_MAKER={}

# constants
ABS_PATH_PRINC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_PATH = ABS_PATH_PRINC + "/data/img/"
PLOT_PATH = ABS_PATH_PRINC + "/data/plots/"
FILE_PATH = ABS_PATH_PRINC + "/data/files/"
LOG_PATH = ABS_PATH_PRINC + "/data/files/log/"
TESTS_PATH = ABS_PATH_PRINC + "/data/tests/"
VIDEO_PATH = ABS_PATH_PRINC + "/data/video/"
TEMPLATE_PATH = ABS_PATH_PRINC + "/data/templates/"

HIERARCHY_TREE = cv2.RETR_CCOMP

CANNY_PERCENTAGE = 1/2

# to plot the changing
BRIGHT_PLOT = []
BRIGHT_PLOT_TIME = 0
LAST_IMGS_MEAN = []
# the minimum and maximum values acceptable for brightness
BRIGHT_MEAN_MIN = 135
BRIGHT_MEAN_MAX = 155

# different convolution matrix
k_contraste = np.array([[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, 5, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]])
k_bords = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, -4, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
kernel_float32 = np.ones((5, 5), np.float32) / 25
kernel_uint8 = np.ones((5, 5), np.uint8)
kernel = np.ones((5,5),np.uint8)

# size of the tag in pixel for different distances
# hardcode
HEIGHT18 = 455 # TODO
WIDTH18 = 350 # TODO
HEIGHT20 = 347
WIDTH20 = 266
HEIGHT30 = 226
WIDTH30 = 175
HEIGHT40 = 172
WIDTH40 = 130
HEIGHT50 = 134
WIDTH50 = 101
HEIGHT65 = 94
WIDTH65 = 73

# used min/max height/width
# distances in cm
DIST_MIN = 20
HEIGHT_MIN = HEIGHT20
WIDTH_MIN = WIDTH20
DIST_MAX = 50
HEIGHT_MAX = HEIGHT50
WIDTH_MAX = WIDTH50

# conversion cm->pixels
CM_PIX = 118.1

THRESH_VALUE = lambda : 135 #2.04*INIT_BRIGHTNESS

def get_template_path():
    size_dir = str(SIZE_X)+"_"+str(SIZE_Y)+"/"
    return TEMPLATE_PATH+size_dir
