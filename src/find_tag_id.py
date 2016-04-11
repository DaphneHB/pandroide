# -*- coding: utf-8 -*-
import numpy as np
from math import fabs
import tools
import cv2


# Robot's direction

def get_direction(direction):
    if direction == 00:
        return "front"
    elif direction == 10:
        return "right"
    elif direction == 01:
        return "left"
    elif direction == 11:
        return "back"


def get_bit_pixel(img, x, y):  # !!!!!!!!!!!!!  ajouter filtre moyenne couleur matrice 9
    val = img[x][y]
    if (val < 127):
        return 1
    else:
        return 0


# define the coordonates of the white area including the 2 children containing datas
def coord_min_max(data_enfant):
    x1 = 99999
    y1 = 99999
    x2 = 0
    y2 = 0
    for coord in enumerate(data_enfant):
        if x1 > coord[1][0]:
            x1 = coord[1][0]
        elif x2 < coord[1][0]:
            x2 = coord[1][0]
        if y1 > coord[1][1]:
            y1 = coord[1][1]
        elif y2 < coord[1][1]:
            y2 = coord[1][1]
    return x1, y1, x2, y2


# extraction of the robot's Id
def extraction_id(img, data_enfant1):
    x1, y1, x2, y2 = coord_min_max(data_enfant1)

    case_length = (x2 - x1) / tools.NB_ID_CASES_SIDE
    initial_point = x1 + case_length / 2
    # print "x1,y1 min : " ,x1 , " ",y1 , " x2,y2 max : " , x2, " " ,y2 , " case_length : ", case_length, ", pointInit: ", initial_point
    idRobot = 0
    for coord in range(tools.NB_ID_CASES_SIDE*tools.NB_ID_CASES_SIDE):
        deltaX = coord % tools.NB_ID_CASES_SIDE
        deltaY = coord / tools.NB_ID_CASES_SIDE # partie entiere
        bitId = get_bit_pixel(img, initial_point + case_length * deltaY, initial_point + case_length * deltaX)
        idRobot = idRobot + np.math.pow(2, coord) * bitId
    #print "idRobot ", idRobot
    return int(idRobot)


# filter the duplicate coordonates defining children corners, return only 4 points
def filter_duplicate(tEnfant):
    filteredList = []
    for ind, e in enumerate(tEnfant):
        x = e[0][0]
        y = e[0][1]
        doublon = False
        for el in enumerate(filteredList):
            if fabs(x - el[1][0]) < 6 and fabs(y - el[1][1]) < 6:
                #    print " doublon!! "
                doublon = True
        if not doublon:
            filteredList.append([x, y])
    return filteredList


# extraction of the robot's direction
def extract_direction(img, data_enfant2):
    x1, y1, x2, y2 = coord_min_max(data_enfant2)
    coordGX = x1 + (x2 - x1) / 4
    coordDX = x2 - (x2 - x1) / 4
    coordMidY = y1 + (y2 - y1) / 2
    # TODO : ne pas prendre uniquemet le pixel du centre, mais la moyenne d'une petite zone
    # applique filtre moyenneur sur le pixel trouver (24 pixels voisins)
    # kernel = np.ones((5,5),np.float32)/25
    # img3 = cv2.filter2D(img2,-1,kernel)
    # print "coordGX ", coordGX , " coordDX ", coordDX, " coordMidY ", coordMidY , " y1 " , y1, " y2 " , y2, " img2 D " , img3[0][coordMidY]
    dirRobot = 10 * get_bit_pixel(img, coordMidY, coordGX) + get_bit_pixel(img, coordMidY, coordDX)
    #print "direction robot ", dirRobot, " Directions : ", get_direction(dirRobot)
    return dirRobot


def separate_children(hierarchy):
    # **[Next, Previous, First_Child, Parent]**
    idParent = idChild1 = idChild2 = -2

    for i, h in enumerate(hierarchy):
        if (h[3] == -1 and h[2] != -1):
            idParent = i
            # print "talbe" ,h , " index ",i
        elif (h[3] == idParent and h[0] != -1):
            idChild2 = i
            # print "table 2er enfant " ,h , " index ",i, " _ ", contours[i]
        elif (h[3] == idParent and h[1] != -1):
            idChild1 = i
            # print "table 1er enfant " ,h , " index ",i, " _ ", contours[i]
    return idChild1, idChild2


# extract id and direction of the robot based on the tag
def obtain_tag_info(img, contours, hierarchy):
    """
    Return the tuple id (int), direction (string)
    """
    idChild1, idChild2 = separate_children(hierarchy[0])
    datasEnfant1 = filter_duplicate(contours[idChild1])
    datasEnfant2 = filter_duplicate(contours[idChild2])
    id = extraction_id(img, datasEnfant1)
    direct = extract_direction(img, datasEnfant2)
    return id,direct

def lecture_tag(tag_found):
    thresh = tools.apply_filters(tag_found)
    # extract hierarchy of the tag
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = obtain_tag_info(tag_found, contours, hierarchy)
    return res

"""
nbCasesIdParCote = 3
import constants as cst
tagName = cst.IMG_PATH+"tag_view1.png"

img2 = cv2.imread(tagName, 0)
# img2 = cv2.imread('testAvecBordure1.png',0) # trainImage
hist, bins = np.histogram(img2.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img3 = cdf[img2]
ret, thresh = cv2.threshold(img3, 127, 255, 0)
# extract hierarchy of the tag
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# faire un find contour sur le tag extrait de la grosse image
t =obtain_tag_info(img2, contours, hierarchy)
print "t = ",t
"""