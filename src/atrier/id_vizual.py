# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:53:28 2016

@author: daphnehb
"""

import cv2

# Grosse marge blanche pour que ca fonctionne "parfaitement"
def getTagBox(gray):
    # find regions of the image that have high horizontal gradients and low vertical gradients
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gray = cv2.equalizeHist(gray)
    median0 = cv2.medianBlur(gray,5)
    (_, gr1) = cv2.threshold(median0, 80, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    (_, gr2) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    close = cv2.morphologyEx(gr1, cv2.MORPH_CLOSE, kernel)
    
    gradX = cv2.Sobel(close, ddepth = cv2.cv.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(close, ddepth = cv2.cv.CV_32F, dx = 0, dy = 1, ksize = -1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    ###
    gradX = cv2.Sobel(gr1, ddepth = cv2.cv.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gr1, ddepth = cv2.cv.CV_32F, dx = 0, dy = 1, ksize = -1)
    # subtract the y-gradient from the x-gradient
    gradient1 = cv2.subtract(gradX, gradY)
    gradient1 = cv2.convertScaleAbs(gradient1)
    median0 = cv2.medianBlur(gradient,5)
    
    dilate0 = cv2.dilate(gradient, None, iterations = 2)
    dilate1 = cv2.dilate(gradient, None, iterations = 1)
    erode0 = cv2.erode(dilate1, None, iterations = 2)
    erode1 = cv2.erode(dilate1, None, iterations = 1)
    # TODO get the best combi
    #equ = cv2.equalizeHist(gray)
    (_, gr) = cv2.threshold(erode0, 220, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(gr, None, iterations = 1)
    blur = cv2.blur(gr,(5,5),0)
    median = cv2.medianBlur(gr,5)
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    #(_, thresh) = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
    #thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 0)
    #blur = cv2.blur(thresh,(5,5),0)   
    #median = cv2.medianBlur(close,5)
    # construct a closing kernel and apply it to the thresholded image
    # used to reveal the rectangular region of the barcode and ignore the rest of the contents of the image
    #closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(median, None, iterations = 1)
    #blur = cv2.blur(closed,(5,5),0)    
    #edges = cv2.Canny(closed.copy(),100,200, L2gradient=True)
    #edges = gradient.copy()
    edges1 = gradient1.copy()
    #closed = cv2.dilate(edges, None, iterations = 1)
    #closed = cv2.erode(closed,None, iterations = 1)
    #v2.imshow("tructhresh",gr1)
    #edges = cv2.Canny(closed.copy(),100,200, L2gradient=True)
    #(contours,hierarchie) = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    (contours1,hierarchie1) = cv2.findContours(edges1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #gray0 = gray.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray1 = gray.copy()
    #cv2.drawContours(gray0, contours, -1, (0, 255, 0), 4)
    #cv2.drawContours(gray1, contours1, -1, (0, 255, 0), 4)
    #cv2.imshow("truc0",gray1)
    contours = contours1
    hierarchie = hierarchie1
    #print "GROSSE HIERARCHIE :",hierarchie
    # if no contours were found, return None
    if len(contours) == 0:
    	return None
    # otherwise, sort the contours by area and compute the rotated
    # the contours with the largest area appear at the front of the list
    # bounding box of the largest contour
    cont = list()
    # for each contour, computing an approximation of this same contour as a polyedre
    for i,cnt in enumerate(contours):
        # approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        # if the shape got convex edges
        if not cv2.isContourConvex(approx) or len(approx)!=4:
            continue
        # if that not a parallelogramm...
        # TODO 
        # if the length is between [10cm;50cm] the original shape
        # TODO
        # if a 30 cm ie taille(pixels)€[100*160;120;170]
        if not isPerimeterOK(cv2.arcLength(approx,True)) :#or not isAreaOK(cv2.contourArea(approx)):
            continue
        # filter contour by 4angles shape
        # filter contour by almost one child
        # no parent?
        if hierarchie[0,i,2]!=-1 and hierarchie[0,i,3]==-1:
            cont.append(approx)
            #cv2.drawContours(gray, [approx], -1, (255, 0, 0), 4)
        #cv2.drawContours(gray, [approx], -1, (0, 0, 255), 4)
    # ATTENTION : making the assumption that the contour with the largest area is the barcoded region of the frame
    # the largest among those with exactly 4 corners and children
    if cont==[]:
        return None
    res = sorted(cont, key = cv2.contourArea, reverse = True)[0]
    
    print "resultat futur box = {}".format(res)
    # take the contour and compute its bounding box 
    # computing the min area bounding/fitting rect (even rotated)
    rect = cv2.minAreaRect(res)
    print "rect interm =",rect
    rect = cv2.cv.BoxPoints(rect)
    print "rect interm2 =",rect
    box = np.int0(rect)
    print "resultat voici la box = {}".format(box)
    
    return res

    
### TODO : between those methods -> substract? is this a real tag?
    

def lecture_tag(tag_found):
    # denoising the image
    equ = cv2.equalizeHist(tag_found)
    #blur = cv2.GaussianBlur(tag_found,(5,5),0)
    # getting only black and white
    closed = cv2.dilate(tag_found, None, iterations = 1)
    
    gradX = cv2.Sobel(closed, ddepth = cv2.cv.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(closed, ddepth = cv2.cv.CV_32F, dx = 0, dy = 1, ksize = -1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    closed = cv2.erode(gradient, None, iterations = 1)
    closed = cv2.dilate(closed, None, iterations = 1)
    (_, thresh0) = cv2.threshold(closed, 170, 255, cv2.THRESH_BINARY_INV)
    (_, thresh1) = cv2.threshold(closed, 170, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(gradient,255,1,1,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    # used to reveal the rectangular region of the barcode and ignore the rest of the contents of the image
    #closed = cv2.morphologyEx(thresh0, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    #closed = cv2.dilate(thresh0, None, iterations = 1)
    #closed = cv2.erode(closed, None, iterations = 3)
    # marking contours
    #edges = cv2.Canny(thresh.copy(),100,200,L2gradient=True)
    # finding all the contours
    contours,hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("gradient",gradient)
    #cv2.imwrite("gradient.png",gradient)
    cv2.imshow("tests",thresh1)
    autre = tag_found.copy()
    cv2.drawContours(autre,contours,-1,(255,0,0),4)
    cv2.imshow("tests1",autre)
    # getting the two main contours (ID and orientation)
    conts = []
    # min size := the size percentage of the orientation block
    min_size = tag_found.size/20
    for i,c in enumerate(contours):
        area = cv2.contourArea(c)
        # testing only the contour bigger than a simple block
        if area>min_size:
            # searching only for 4corners  polys
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.04*peri,True)
            if len(approx)==4:
                # getting only contours 
                # with a previous or(exclusive) a next
                # and who's got a parent
                if hierarchy[0,i,3]!=-1:#(bool(hierarchy[0,i,1]!=-1)!=bool(hierarchy[0,i,0]!=-1)) \
                #and hierarchy[0,i,3]!=-1:
                    # adding it to the important ones
                    i = i
                    parent = hierarchy[0,i,3]
                    print "contour parent de  {} : {}".format(i,hierarchy[0,parent])
                conts.append(c)
                    #conts.append(contours[parent])
                # end if
            # end if
        # end if
    # end for
    cv2.drawContours(tag_found, conts, -1, (0, 255, 0), 4)        
    
    cv2.imshow("ma lect",tag_found)
    #return (identity,orientation)

