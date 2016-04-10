# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:51:37 2016

@author: emilie
"""

import cv2
import numpy as np

img = cv2.imread('lena.jpg')

res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

#OR

height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
res.show()