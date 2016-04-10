# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# import the necessary packages
import numpy as np
import cv2
from scipy import optimize


FNAME = "tag5.jpg"
tag_seul = cv2.imread("tag.png",0)


### FUNCTIONS DEFINITIONS
def testImgFile(filename):
    # load the image and convert it to grayscale
    image = cv2.imread(filename)
    image = cv2.resize(image,(500,500))
    
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #test = matchingTemplate(tag_seul,gray)
    tag_found = foundTagImg(gray)
    if tag_found is None:
        tag_found = gray
            
    # Display the resulting frame
    cv2.imshow("grad",tag_found)
    #cv2.imshow("gray",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def takeVideo():
    cap = cv2.VideoCapture(0)
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #test = matchingTemplate(tag_seul,gray)
        tag_found = foundTagImg(gray)
        if tag_found is None:
            tag_found = gray
        # Display the resulting frame
        cv2.imshow('frame',frame)
        cv2.imshow('frameGrad',tag_found)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
"""
Getting the initial gray image
Returning the simple tag found in this initial image
"""
def foundTagImg(gray):
    # getting the corresponding tag box in complex image
    box = getTagBox(gray)
    if (box is None):
        return None
    # getting a simple image of the tag (only)
    #tag_gray = imgHomot(gray,box)
    tag_gray = flatten_tag(gray,box)
    # reading the info in the tag
    #lecture_tag(tag_gray)
    return tag_gray
    
# TODO better match template
def matchingTemplate(img_init,img_in_scene):
    # match template
    w,h = img_init.shape[::-1]
    img_test = img_in_scene.copy()
    res = cv2.matchTemplate(img_test,img_init,cv2.TM_CCOEFF_NORMED)
    minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(res)
    top_left = maxLoc
    bottom_right = (top_left[0]+w,top_left[1]+h)
    cv2.rectangle(img_test,top_left,bottom_right,255,2)
    return img_test

# Grosse marge blanche pour que ca fonctionne "parfaitement"
def getTagBox(gray):
    # find regions of the image that have high horizontal gradients and low vertical gradients
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 0, dy = 1, ksize = -1)
    
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    # TODO get the best combi
    #blurred = cv2.blur(gradient,(5,5),0)
    #(_, thresh) = cv2.threshold(gradient, 220, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    # used to reveal the rectangular region of the barcode and ignore the rest of the contents of the image
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 1)
    closed = cv2.dilate(closed, None, iterations = 1)
    edges = cv2.Canny(closed,100,200, L2gradient=True)
    
    (contours,hierarchie) = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
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
        # filter contour by rectangular shape
        # filter contour by almost one child
#        cv2.drawContours(gray, [approx], -1, (0, 255, 0), 4)
        # no parent?
        if len(approx)==4 and hierarchie[0,i,2]!=-1 and hierarchie[0,i,3]==-1:
            cont.append(approx)
    # ATTENTION : making the assumption that the contour with the largest area is the barcoded region of the frame
    # the largest among those with exactly 4 corners and children
    res = sorted(cont, key = cv2.contourArea, reverse = True)[0]
    #cv2.drawContours(gray, [res], -1, (0, 255, 0), 4)
    
    # take the contour and compute its bounding box 
    # computing the min area bounding/fitting rect (even rotated)
    rect = cv2.minAreaRect(res)
    box = np.int0(cv2.cv.BoxPoints(rect))
        
    return box
    
    
def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)
 
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
     
    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
  
    return hnew
        
def imgHomot(gray,box):
    box = rectify(box)
    h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
    retval = cv2.getPerspectiveTransform(box,h)
    warp = cv2.warpPerspective(gray,retval,(450,450))
    return warp
   

"""
box : the rect contour of the tag
"""        
def flatten_tag(gray,box):
    # Create a mask of the label
    mask = np.zeros(gray.shape,np.uint8)
    cv2.drawContours(mask, [box],0,255,-1)
    # Find the 4 borders
    ddepth = cv2.CV_8U
    borderType=cv2.BORDER_DEFAULT
    left = cv2.Sobel(mask,ddepth,1,0,ksize=1,scale=1,delta=0,borderType=borderType)
    right = cv2.Sobel(mask,ddepth,1,0,ksize=1,scale=-1,delta=0, borderType=borderType)
    top = cv2.Sobel(mask,ddepth,0,1,ksize=1,scale=1,delta=0,borderType=borderType)
    bottom = cv2.Sobel(mask,ddepth,0,1,ksize=1,scale=-1,delta=0,borderType=borderType)
    
    # Remove noise from borders
    kernel = np.ones((2,2),np.uint8)
    left_border = cv2.erode(left,kernel,iterations = 1)
    right_border = cv2.erode(right,kernel,iterations = 1)
    top_border = cv2.erode(top,kernel,iterations = 1)
    bottom_border = cv2.erode(bottom,kernel,iterations = 1)
    # Equations 1 and 2: c1 + c2*x + c3*y + c4*x*y, c5 + c6*y + c7*x + c8*x^2
    # Find coeficients c1,c2,c3,c4,c5,c6,c7,c8 by minimizing the error function. 
    # Points on the left border should be mapped to (0,anything).
    # Points on the right border should be mapped to (108,anything)
    # Points on the top border should be mapped to (anything,0)
    # Points on the bottom border should be mapped to (anything,70)
    sum_of_squares_y = '+'.join( [ "(c[0]+c[1]*%s+c[2]*%s+c[3]*%s*%s)**2" % \
        (x,y,x,y) for y,x,z in np.transpose(np.nonzero(left_border)) ])
    sum_of_squares_y += " + "
    sum_of_squares_y += '+'.join( [ "(-108+c[0]+c[1]*%s+c[2]*%s+c[3]*%s*%s)**2" % \
        (x,y,x,y) for y,x,z in np.transpose(np.nonzero(right_border)) ])
    res_y = optimize.minimize(lambda c: eval(sum_of_squares_y),(0,0,0,0),method='SLSQP')
    
    sum_of_squares_x = '+'.join( [ "(-70+c[0]+c[1]*%s+c[2]*%s+c[3]*%s*%s)**2" % \
        (y,x,x,x) for y,x,z in np.transpose(np.nonzero(bottom_border)) ] )
    sum_of_squares_x += " + "
    sum_of_squares_x += '+'.join( [ "(c[0]+c[1]*%s+c[2]*%s+c[3]*%s*%s)**2" % \
        (y,x,x,x) for y,x,z in np.transpose(np.nonzero(top_border)) ] )
    res_x = optimize.minimize(lambda c: eval(sum_of_squares_x),(0,0,0,0), method='SLSQP')
    
    flattened = np.zeros(gray.shape, gray.dtype) 
    for y,x,z in np.transpose(np.nonzero(mask)):
        new_y = map_y(res_x.x,[y,x]) 
        new_x = map_x(res_y.x,[y,x])
        flattened[float(new_y)][float(new_x)] = gray[y][x]
    # Crop the image 
    flattened = flattened[0:70, 0:105]
    return flattened

# INTER FUNCTIONS
# Map the image using equatinos 1 and 2 (coeficients c1...c8 in res_x and res_y)
def map_x(res, coord):
    return res[0] + res[1]*coord[1] + res[2]*coord[0] + res[3]*coord[1]*coord[0]
def map_y(res, coord):
    return res[0] + res[1]*coord[0] + res[2]*coord[1] + res[3]*coord[1]*coord[1]

    
def lecture_tag(tag_found):
    # denoising the image
    #equ = cv2.equalizeHist(tag_found)
    blur = cv2.GaussianBlur(tag_found,(5,5),0)
    # getting only black and white
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    # used to reveal the rectangular region of the barcode and ignore the rest of the contents of the image
    #closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    #closed = cv2.dilate(closed, None, iterations = 1)
    #closed = cv2.erode(closed, None, iterations = 1)
    # marking contours
    edges = cv2.Canny(thresh,100,200,L2gradient=True)
    # finding all the contours
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    # getting the two main contours (ID and orientation)
    conts = []
    # min size := the size percentage of the orientation block
    min_size = thresh.size/5
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
                    #conts.append(c)
                    conts.append(contours[parent])
                # end if
            # end if
        # end if
    # end for
    cv2.drawContours(tag_found, conts, -1, (0, 255, 0), 4)        
    
    cv2.imshow("ma lect",thresh)
    #return (identity,orientation)

### TESTS

#takeVideo()
testImgFile(FNAME)
