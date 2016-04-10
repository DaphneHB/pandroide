# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 18:23:22 2016

@author: daphnehb
"""

import cv2

FILENAME = "newtag03L.png"

img = cv2.imread(FILENAME)
_,inverse = cv2.threshold(img,190,255,cv2.THRESH_BINARY_INV)
cv2.imwrite("inv_"+FILENAME,inverse)