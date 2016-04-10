# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:19:27 2016

@author: daphnehb
"""

import numpy as np
import cv2
from PyQt4 import QtGui,QtCore

DATA_PATH = "../data/"

class WindowPrinc(QtGui.QMainWindow):
    
    def __init__(self,parent=None):
        super(WindowPrinc,self).__init(parent=parent)
        self.initUI()
        
    def initUI(self):
        pass
    