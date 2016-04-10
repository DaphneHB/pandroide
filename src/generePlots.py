# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:53:58 2016

@author: daphnehb & emilie
"""
import tools
import matplotlib.pyplot as plt
import plotly.plotly as py


def boxPlotBestTime():
    """
    Getting the box plots with outsider
    Testing with our ex-filters, elias's and improved elias's
    """
    if tools.TPS_ELIAS==[] or tools.TPS_EMILIAS==[] or tools.TPS_NOUS==[]:
        print "Boxplots comparing algorithms not generated"
        return None
    plt.figure()
    plt.boxplot([tools.TPS_NOUS,tools.TPS_ELIAS,tools.TPS_EMILIAS])
    plt.xticks([1,2,3],["Our","Elias","Emilias"])
    plt.title("Comparison of filters used (boxplots")
    plt.savefig(tools.PLOT_PATH+'diff_filters_box.png')
    #plt.show()

def plotBestTime():
    """
    Getting the curve plots with outsider
    Testing with our ex-filters, elias's and improved elias's
    """
    if tools.TPS_ELIAS==[] or tools.TPS_EMILIAS==[] or tools.TPS_NOUS==[]:
        print "Curve plots comparing algorithms not generated"
        return None
    plt.figure()
    plt.plot(tools.TPS_NOUS)
    plt.plot(tools.TPS_ELIAS)
    plt.plot(tools.TPS_EMILIAS)
    plt.title("Comparison of filters used (plots)")
    plt.legend(["Our","Elias","Emilias"], loc='upper left')
    plt.savefig(tools.PLOT_PATH+'diff_filters_curves.png')
    #plt.show()

def results_pie_chart():
    if tools.TRUE_NEG==[] or tools.TRUE_POS==[] or tools.FALSE_NEG==[] or tools.FALSE_POS:
        print "Pie chart of results not generated"
        return None
    # The slices will be ordered and plotted counter-clockwise.
    labels = 'True positives', 'True negatives', 'False positives', 'False negatives'
    sizes = [tools.TRUE_POS,tools.TRUE_NEG,tools.FALSE_POS,tools.FALSE_NEG]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0.1, 0.1, 0, 0)  # only "explode" the 1st and 2nd slice

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')

    plt.savefig(tools.PLOT_PATH+"results_percentages.png")
    plt.show()