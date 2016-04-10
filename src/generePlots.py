# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:53:58 2016

@author: 3200234
"""
import tools
import matplotlib.pyplot as plt

def boxPlotBestTime():
    """
    Getting the box plots with outsider
    Testing with our ex-filters, elias's and improved elias's
    """
    plt.figure()
    plt.boxplot([tools.TPS_NOUS,tools.TPS_ELIAS,tools.TPS_EMILIAS])
    plt.xticks([1,2,3],["Our","Elias","Emilias"])
    plt.title("Comparison of filters used (boxplots")
    plt.savefig(tools.PLOT_PATH+'diff_filters_box.png')
    #plt.show()

def boxPlotBestTimeSans():
    """
    Getting the box plots without outsiders
    Testing with our ex-filters, elias's and improved elias's
    """
    plt.figure()
    plt.boxplot([tools.TPS_NOUS,tools.TPS_ELIAS,tools.TPS_EMILIAS],0,'')
    plt.xticks([1,2,3],["Our","Elias","Emilias"])
    plt.title("Comparison of filters used (boxplots")
    plt.savefig(tools.PLOT_PATH+'diff_filters_box_sans.png')
    #plt.show()

def plotBestTime():
    """
    Getting the curve plots with outsider
    Testing with our ex-filters, elias's and improved elias's
    """
    plt.figure()
    plt.plot(tools.TPS_NOUS)
    plt.plot(tools.TPS_ELIAS)
    plt.plot(tools.TPS_EMILIAS)
    plt.title("Comparison of filters used (plots)")
    plt.legend(["Our","Elias","Emilias"], loc='upper left')
    plt.savefig(tools.PLOT_PATH+'diff_filters_curves.png')
    #plt.show()
