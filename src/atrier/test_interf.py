# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 17:06:41 2016

@author: daphnehb
"""


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt4 import QtGui,QtCore
import sys
import cv2

import random

class Window(QtGui.QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()
        ### lecture video
        self.camera = cv2.VideoCapture(0)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        #self.button = QtGui.QPushButton('Plot')
        #self.button.clicked.connect(self.plot)

        # set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        #layout.addWidget(self.button)
        self.setLayout(layout)
        
    def show_img(self):    
        _,frame = self.camera.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.plot(gray)
        
            
    def closeEvent(self,event):
        # When everything done, release the capture
        self.camera.release()
        event.accept()

    def keyPressEvent(self, e):
        
        if e.key() == QtCore.Qt.Key_Q or e.key()==QtCore.Qt.Key_Escape:
            self.close()
    
    def plot(self,img):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.hold(False)

        # plot data
        ax.plot(img, '*-')

        # refresh canvas
        self.canvas.draw()
    
def lancementVideo(mainWindow):
    #cap = cv2.Video
    pass

def plot_imshow(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())