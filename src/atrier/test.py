# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 01:08:37 2016

@author: emilie
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
 
 
#...........................Importation de l'image:....................
 
img = cv2.imread('tag.png')
imgray = cv2.cvtColor (img,cv2.COLOR_BGR2GRAY) #niveau de gris
 
#.............................detection de contours.....................
 
ret,thresh = cv2.threshold(imgray,127,255,0)
# image,contours and hierarchy
image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
len(contours)#Comptage du nombre de contours fermés
cnt = len(contours)
cv2.drawContours(img,contours,-1,(0,255,0),2)
 
#............................... Affichage...............................
 
cv2.imshow('Contours',img)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
 
#....................filtre de sobel...................................
 
 
scale = 1
delta = 0
ddepth = cv2.CV_16S
 
img = cv2.imread('Cap.jpg')
img = cv2.GaussianBlur(img,(3,3),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
#Calcul du gradiant
# Gradient-X
grad_x = cv2.Sobel(gray,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
#grad_x = cv2.Scharr(gray,ddepth,1,0)
 
# Gradient-Y
grad_y = cv2.Sobel(gray,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
#grad_y = cv2.Scharr(gray,ddepth,0,1)
 
abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
abs_grad_y = cv2.convertScaleAbs(grad_y)
 
sobel = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
#dst = cv2.add(abs_grad_x,abs_grad_y)
 
cv2.imshow('image sobel',sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite ("resultat .jpg",sobel)
##.............................Squelettisation ..........................
dst = sobel
img = cv2.imread('Cap.jpg',0)
size = np.size(dst)
skel = np.zeros(dst.shape,np.uint8)
  
ret,dst = cv2.threshold(dst,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
  
while( not done):
    eroded = cv2.erode(dst,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(dst,temp)
    skel = cv2.bitwise_or(skel,temp)
    dst = eroded.copy()
  
    zeros = size - cv2.countNonZero(dst)
    if zeros==size:
        done = True
  
cv2.imshow("skel",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
cv2.imwrite ("skel.jpg",skel)
#................................test de dilatation........................
import pymorph
from PIL import Image
 
#skel = sobel
size = np.size(skel)
skel = np.zeros(skel.shape,np.uint8)
 
ret,skel = cv2.threshold(skel,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
skel = 255 - skel
skel = cv2.dilate(skel, element, iterations=3)
cv2.imshow("dilatation",skel)
cv2.waitKey(0)
cv2.imwrite ("dilatation.jpg",skel)
#............................... Nombre de pixels......................
#print edges          ##edges est une matrice          
x = len (dst[0])
print len (dst[0])    ##la taille de chaque ligne cette la matrice
                            #correspond à la largeur de image
y = len (dst)
print len (dst)       #le nombre de lignes de cette matrice
                           #correspond à la hauteur de l'image
 
#donc chaque entrée edges[x][y] correspond à la valeur d'un pixel
# 0 pour le noir et 255 pour blanc
#avec x correspond à chaque pixel différent de 0, donc 255 puisque j'ai une image binaire
 
 
#print np.transpose (np.nonzero(edges))[x] ..
 
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,contours,-1,(0,255,0),1)
pixelpoints = np.transpose(np.nonzero(mask))
print (pixelpoints)
 
#................................Valeurs maximale et minimale..............
 
min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(imgray,mask = mask)
print (min_val, max_val,min_loc, max_loc)
 
#......................... points extremes.....................
 
x = cnt[:]
int (x)
x_min_loc = x.argmin()
point = cnt[x_min_loc]
leftmost = tuple(point[0])
 
x_max_loc = x.argmax()
point = cnt[x_max_loc]
 
#........exportation des pixels vers un fichier .txt........................
f =  open("data.txt", 'w')
f.write('cnt' +repr (x, y))
f.close()