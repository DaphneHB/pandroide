# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

@author : daphnehb & emilie
"""

import test_pc_detect as td
import generePlots as plt

""" Testing the robot's launch """
td.takeVideo()
#td.testImgFile(FNAME)

""" Trying to re-generate and save the plot corresponding to what was tested """
plt.plotBestTime()
plt.boxPlotBestTime()
plt.results_pie_chart()