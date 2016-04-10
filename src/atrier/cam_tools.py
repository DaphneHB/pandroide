# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:23:42 2016

@author: daphnehb
"""

import cv2
import picamera
import error_tools as err


""" Number of image per seconds (5 by default)"""
FRAMERATE = 5

def setFramerate(fps):
    global FRAMERATE
    FRAMERATE = fps
        

"""
A simple class to access the camera stream
Source independent
"""
class MyCam:
    """
    Creating the Object
    if raspberry arg is True -> activating raspberry picamera
    """
    def __init__(self,raspberry):
        self.ON_RASPB = raspberry
        self.camera = self.activateCamera()
    
    """
    Activating the camera (Webcam for a pc)
    if raspberry arg is True -> activating raspberry picamera
    
    Returning the cam obj
    """
    def activateCamera(self):
        global FRAMERATE
        if self.ON_RASPB:
            camera = picamera.PiCamera(framerate=FRAMERATE)            
        else:
            camera = cv2.VideoCapture(0)
            cap.set(cv2.cv.CV_CAP_PROP_FPS, FRAMERATE)
        return camera
    
    """
    Taking a picture of the current camera in arg
    
    Return the opencv capture
    """
    def takeCapture(self):
        # in case the camera was shutdown
        if self.camera is None:
            # TODO : raising exception?            
            #raise 
            # or just print a warning and activating the camera?
            err.print_err("The camera was shutdown\nHad to re-activate it! ><")
            self.activateCamera()
            
        if self.ON_RASPB:
            # raspberry case
            self.camera.capture("image.png")
            frame = cv2.imread("image.png")
        else:
            # computer case
            _,frame = self.camera.read()
        return frame
       
    """
    Close the current camera
    Not necessar on the raspberry
    """
    def shutdown(self):
        if self.camera is None:
            return None
        if self.ON_RASPB:
            self.camera.close()
        else:
            self.camera.release()
            
    def changeBrightness(self):
        pass
    # TODO methods to set camera's properties
        
    """
    Changing the global value of Framerate
    Changing the current camera's fps
    """
    def changeFramerate(self,fps):
        global FRAMERATE
        FRAMERATE = fps
        # changing the framerate of the current camera
        if self.ON_RASPB:
            self.camera._set_camera_mode(framerate=fps)
        else:
            self.camera.set(cv2.cv.CV_CAP_PROP_FPS, fps)
