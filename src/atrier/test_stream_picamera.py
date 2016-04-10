# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:38:09 2016

@author: daphnehb
"""

import time
import picamera
import picamera.array
import cv2

with picamera.PiCamera() as camera:
    camera.start_preview()
    time.sleep(2)
    with picamera.array.PiRGBArray(camera) as stream:
        camera.capture(stream, format='bgr')
        # At this point the image is available as stream.array
        image = stream.array
