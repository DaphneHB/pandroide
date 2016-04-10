# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:30:18 2016

@author: emilie
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 01:04:35 2016

@author: emilie
"""

import cv2
import numpy as np
from math import fabs
import time

img = cv2.imread('tag_dirNN.png',0)


nbCasesIdParCote = 3
dtime =time.time()
# equialisation de l histogramme
# https://en.wikipedia.org/wiki/Histogram_equalization
# http://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html#gsc.tab=0

#hist,bins = np.histogram(img2.flatten(),256,[0,256])
#cdf = hist.cumsum()
#cdf_normalized = cdf * hist.max()/ cdf.max()
#cdf_m = np.ma.masked_equal(cdf,0)
#cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
#cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cv2.equalizeHist(img) #cdf[img2]
#print cdf
#blur = cv2.GaussianBlur(img2,(5,5),0)
ret,thresh = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#ret,thresh = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresh,100,200,L2gradient=True)

contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imwrite('tag_find-contour.png',thresh)


def pixImg(img):
    listImg = list(img)
    tabImg = np.asarray(map(str,listImg))
    #tabImg.reshape(img.shape)
    return tabImg
    
#print contours
# **[Next, Previous, First_Child, Parent]**
try:
    fichier = open("findCont.txt","w")
    fichier.write("Mon image\n")
    #fichier.write(pixImg(thresh))
    fichier.write("Hierarchie\n")
    fichier.write(str(hierarchy))
    fichier.write("\nContours\n")
    fichier.write(str(contours))
    fichier.close()
except IOError:
    pass

#print hierarchy
#print contours
idParent = idEnfant1= idEnfant2 = -2
for i,h in enumerate(hierarchy[0]) :
    if (h[3] == -1 and h[2] != -1) :
        idParent = i 
        print "table" ,h , " index ",i
    elif(h[3] == idParent  and h[0] != -1) :
        idEnfant2 = i
        print "table 2d child " ,h , " index ",i, " _ ", contours[i]    
    elif(h[3] == idParent  and h[1] != -1) :
        idEnfant1 = i        
        print "table 1st child " ,h , " index ",i, " _ ", contours[i]


#analyseEnfant2(idEnfant2,contours[idEnfant2])


#def analyseEnfant2( idEnfant1):


def filtreDoublon(tEnfant) :
    #print " len tEnfant ", len(tEnfant)," len tEnfant[0]", tEnfant[0]," len tEnfant[1]", tEnfant[1]
    filteredList = []
    for ind, e in enumerate(tEnfant) :
        x=e[0][0]
        y=e[0][1] 
        #print  " ------- analysing ---", x,",",y 
        doublon = False

        for el in enumerate(filteredList) :
           # print " compared with ", el, " X=", el[1][0], " Y=", el[1][1]
            if fabs(x - el[1][0]) < 6 and fabs(y - el[1][1]) < 6:
            #    print " doublon!! "
                doublon = True
            
        if not doublon:
            filteredList.append([x,y])            
    return filteredList 

#filtreDoublon(contours[idEnfant2])    
datasEnfant1 = filtreDoublon(contours[idEnfant1])
print "datas enfant1 " , datasEnfant1
datasEnfant2 = filtreDoublon(contours[idEnfant2])
print "datas enfant2 " , datasEnfant2

def getBitPixel(x,y):           # !!!!!!!!!!!!!  ajouter filtre moyenne couleur matrice 9
    val = img2[x][y]
    if(val<127):
        return 1
    else:
        return 0

def coordMinMax(datasEnfant):
    x1 = 99999
    y1 = 99999
    x2 = 0
    y2 = 0
    for coord in enumerate(datasEnfant) :
        if x1 > coord[1][0] :
            x1 = coord[1][0]
        elif x2 < coord[1][0] :
            x2 = coord[1][0]
        if y1 > coord[1][1] :
            y1 = coord[1][1]
        elif y2 < coord[1][1] :
            y2 = coord[1][1]
    return x1,y1,x2,y2

def extractionId(datasEnfant1):
    x1,y1,x2,y2 = coordMinMax(datasEnfant1)
    # affichage coint haut gauche
    subtab = edges[y1-100:y1+100,x1-100:x1+100]
    cv2.imshow("bord",subtab)
    cv2.waitKey(0)
    cv2.destroyWindow("bord")
    tailleCase = (x2-x1)/nbCasesIdParCote
    pointInitial = x1 + tailleCase/2
    print "x1,y1 min : " ,x1 , " ",y1 , " x2,y2 max : " , x2, " " ,y2 , " tailleCase : ", tailleCase, ", pointInit: ", pointInitial    
    idRobot = 0
    for coord in range(nbCasesIdParCote * nbCasesIdParCote) :
        deltaX = coord % nbCasesIdParCote
        deltaY = coord / nbCasesIdParCote   # partie entiere
       # coordColor.append({(pointInitial+tailleCase*deltaY,pointInitial+tailleCase*deltaX):getCouleurPixel(pointInitial+tailleCase*deltaY,pointInitial+tailleCase*deltaX) })
        #coordColorValue[deltaY,deltaX]= pointInitial+tailleCase*deltaX +tailleCase*deltaY
        bitId = getBitPixel(pointInitial+tailleCase*deltaY,pointInitial+tailleCase*deltaX)
        #img2[pointInitial+tailleCase*deltaY][pointInitial+tailleCase*deltaX]=45    
        idRobot = idRobot + np.math.pow(2,coord)*bitId
    print " idRobot ",idRobot
    
    
    
        
extractionId(datasEnfant1)

def extractionDirection(datasEnfant2):
    x1,y1,x2,y2 = coordMinMax(datasEnfant2)
    coordGX = x1 + (x2 - x1)/4 
    coordDX = x2 - (x2 - x1)/4
    coordMidY = y1 + (y2 - y1)/2
    print "coordGX ", coordGX , " coordDX ", coordDX, " coordMidY ", coordMidY , " y1 " , y1, " y2 " , y2, " img2 D " , img2[0][coordMidY]
    dirRobot = 10 * getBitPixel(coordMidY, coordGX) + getBitPixel(coordMidY, coordDX)
    print "direction robot ", dirRobot
    img2[coordMidY][coordDX]=255    
    img2[coordMidY][coordDX-1]=255    
    img2[coordMidY][coordDX-2]=255    
    img2[coordMidY][coordDX-3]=255    

    img2[coordMidY][coordGX]=255  
    img2[coordMidY][coordGX-1]=255    
    img2[coordMidY][coordGX-2]=255    
    img2[coordMidY][coordGX-3]=255  

    cv2.imshow("test",img2)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    
extractionDirection(datasEnfant2)
ftime = time.time()
print("Temps total: {}s".format(ftime-dtime))
    
