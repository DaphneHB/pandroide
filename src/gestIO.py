# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:28:54 2016

@author: 3200234
"""

import os,tools
import time, datetime
import threading

def writeOutputFile(myStr, directory=None, path=None, log=False):
    """
    Write the times computed
    """
    extension = ".txt" if not log else ".log"
    if directory is None:
        directory = tools.FILE_PATH if not log else tools.LOG_PATH		
    if path is None:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mmin')
        filename = st+""+extension
        path = directory+""+filename
    else:
        directory = os.path.abspath(os.path.dirname(path))
    # creating the directory if not existing
    if not os.path.exists(directory):
        os.makedirs(directory)
    # overriding the previous one (if existing)
    with open(path,'w') as monfile:
        monfile.write(myStr+"\nEnd vizualization\n")

class WriteLog(threading.Thread):
    def __init__(self, filename):
        threading.Thread.__init__(self)
        self._stop = False
        self.daemon = True
