# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import fabs
from enum import Enum


nbCasesIdParCote = 3
tagName = 'tag_dirNN.png'
img = cv2.imread(tagName,0)
# Robot's direction
class Directions(Enum) :
    front = 00
    right = 10
    left = 01
    back = 11
    
    def __str__(self):
        return str(self.name)


def getBitPixel(x,y):           # !!!!!!!!!!!!!  ajouter filtre moyenne couleur matrice 9
    val = img[x][y]
    if(val<127):
        return 1
    else:
        return 0

#define the coordonates of the white area including the 2 children containing datas
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

#extraction of the robot's Id
def extractionId(datasEnfant1):
    x1,y1,x2,y2 = coordMinMax(datasEnfant1)
        
    tailleCase = (x2-x1)/nbCasesIdParCote
    pointInitial = x1 + tailleCase/2
    #print "x1,y1 min : " ,x1 , " ",y1 , " x2,y2 max : " , x2, " " ,y2 , " tailleCase : ", tailleCase, ", pointInit: ", pointInitial    
    idRobot = 0
    for coord in range(nbCasesIdParCote * nbCasesIdParCote) :
        deltaX = coord % nbCasesIdParCote
        deltaY = coord / nbCasesIdParCote   # partie entiere
        bitId = getBitPixel(pointInitial+tailleCase*deltaY,pointInitial+tailleCase*deltaX)
        idRobot = idRobot + np.math.pow(2,coord)*bitId
    print "idRobot ",idRobot

#filter the duplicate coordonates defining children corners, return only 4 points 
def filtreDoublon(tEnfant) :
    filteredList = []
    for ind, e in enumerate(tEnfant) :
        x=e[0][0]
        y=e[0][1] 
        doublon = False
        for el in enumerate(filteredList) :
            if fabs(x - el[1][0]) < 6 and fabs(y - el[1][1]) < 6:
            #    print " doublon!! "
                doublon = True
        if not doublon:
            filteredList.append([x,y])            
    return filteredList 

# extraction of the robot's direction
def extractionDirection(datasEnfant2):
    x1,y1,x2,y2 = coordMinMax(datasEnfant2)
    coordGX = x1 + (x2 - x1)/4 
    coordDX = x2 - (x2 - x1)/4
    coordMidY = y1 + (y2 - y1)/2
# TODO : ne pas prendre uniquemet le pixel du centre, mais la moyenne d'une petite zone
#applique filtre moyenneur sur le pixel trouver (24 pixels voisins)
    #kernel = np.ones((5,5),np.float32)/25
    #img3 = cv2.filter2D(img,-1,kernel)
    #print "coordGX ", coordGX , " coordDX ", coordDX, " coordMidY ", coordMidY , " y1 " , y1, " y2 " , y2, " img2 D " , img3[0][coordMidY]
    dirRobot = 10 * getBitPixel(coordMidY, coordGX) + getBitPixel(coordMidY, coordDX)
    print "direction robot ", dirRobot, " Directions : ", Directions(dirRobot)
    
def separatingChildren(hierarchy) :
    # **[Next, Previous, First_Child, Parent]**
    idParent = idChild1= idChild2 = -2
    
    for i,h in enumerate(hierarchy) :
        if (h[3] == -1 and h[2] != -1) :
            idParent = i 
            #print "talbe" ,h , " index ",i
        elif(h[3] == idParent  and h[0] != -1) :
            idChild2 = i
                #print "table 2er enfant " ,h , " index ",i, " _ ", contours[i]    
        elif(h[3] == idParent  and h[1] != -1) :
            idChild1 = i        
                    #print "table 1er enfant " ,h , " index ",i, " _ ", contours[i]
    return idChild1,idChild2
    
# extract id and direction of the robot based on the tag
def tagInfos() :
    

    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    ret,thresh = cv2.threshold(img2,127,255,0)

    # extract hierarchy of the tag
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    idChild1,idChild2 = separatingChildren(hierarchy[0])
    datasEnfant1 = filtreDoublon(contours[idChild1])
    datasEnfant2 = filtreDoublon(contours[idChild2])
    extractionId(datasEnfant1)    
    extractionDirection(datasEnfant2)


tagInfos()