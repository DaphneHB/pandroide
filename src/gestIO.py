# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:28:54 2016

@author: 3200234
"""

import os,tools
import time, datetime

def writeOutputFile(myStr):
    """
    Write the times computed
    """
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mmin%Ssec')
    filename = st+".txt"
    if not os.path.exists(tools.FILE_PATH):
        os.makedirs(tools.FILE_PATH)
    path = tools.FILE_PATH+""+filename
    # overriding the previous one (unexisting)
    with open(path,'w') as monfile:
        monfile.write(myStr+"\nFin Prise\n")
