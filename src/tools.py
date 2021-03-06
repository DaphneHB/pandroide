# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:21:36 2016

@author: daphnehb
"""

import time
import find_tag_id as tagid
# to make files more legible
from constants import *

def orb_test(img):
    orb = cv2.ORB()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    ret = cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0)
    return ret
    
# differents effets et filtres
def use_histo(img):
    """
    :param img:
    :return: the equlization by histogram found
    """
    return cv2.equalizeHist(img)

def check_change_brightness(list_means):
    global BRIGHTNESS_MIN, BRIGHTNESS_MAX, BRIGHTNESS_COMP, BRIGHTNESS_STEP, INIT_BRIGHTNESS
    BRIGHT_PLOT.append(INIT_BRIGHTNESS)
    print "Checking brightness with mean images :",list_means
    tab_min = [x<BRIGHT_MEAN_MIN for x in list_means]
    where_min = tab_min.count(True)
    # if there are too many images under the thresh intervalle
    change = 0
    if where_min>=BRIGHT_MIN_WRONG_IMG:
        change = BRIGHTNESS_STEP
        print "Increasing the brightness of the camera: +{}".format(change)
    else:
        tab_max = [x>BRIGHT_MEAN_MAX for x in list_means]
        where_max = tab_max.count(True)
        # if there are too many images over the thresh intervallle
        if where_max>=BRIGHT_MIN_WRONG_IMG:
            change = -BRIGHTNESS_STEP
	    print "Reducing the brightness of the camera : {}".format(change)
    # re-setting the brightness
    INIT_BRIGHTNESS += change
    return change

def verify_brightness(image, go=False):
    global LAST_IMGS_MEAN, BRIGHTNESS_COMP
    n_mean = cv2.mean(image)[0]
    # more accurate & faster : n_mean = image.mean()
    if len(LAST_IMGS_MEAN)==BRIGHTNESS_COMP:
        LAST_IMGS_MEAN.pop(0)
        LAST_IMGS_MEAN.append(n_mean)
    else:
        LAST_IMGS_MEAN.append(n_mean)
    # if we do apply the verification
    change = 0
    if go:
    	change = check_change_brightness(LAST_IMGS_MEAN)
    return change

def houghlines(img):
    hough = img.copy()
    lines = cv2.HoughLines(hough, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imwrite("houghlinesImg.png",hough)
    return hough


def find_contours(img_copy):
    """
    :type img_copy: np.ndarray -> an opencv img
    """
    global HIERARCHY_TREE
    contours, hierarchy = cv2.findContours(img_copy, HIERARCHY_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def exec_canny(img_copy):
    """
    # on marque les contours
    :type img_copy: np.ndarray -> an opencv img
    """
    edges = cv2.Canny(img_copy, 100, 200, L2gradient=True)
    return edges


def color_gray_img(img, to_gray=True):
    if to_gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def copy(img):
    return img.copy()


# meilleur : seuillage+otsu+closingMorpho
"""
Black-White threshold
"""


def thresholding(img, seuil=THRESH_VALUE(), reverse=False, with_otsu=False, adaptative=False):
    if reverse:
        val = cv2.THRESH_BINARY_INV
    else:
        val = cv2.THRESH_BINARY
    if with_otsu:
        val += cv2.THRESH_OTSU
    if adaptative:
        _, thresh = cv2.adaptativeThreshold(img, seuil, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, val, 45, 0)
    else:
        _, thresh = cv2.threshold(img, seuil, 255, val)
    return thresh



def convoluer(img, mat=k_contraste):
    # equivalent de dst
    return cv2.filter2D(img, -1, mat)


def gradientSobelXY(img):
    gradX = cv2.Sobel(img, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(img, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    return gradient

"""
kind = 0:simple blur; 1:median blur; 2:gaussian blur
Default : simple blur
"""
def blurring(img, taille_kernel, kind=0):
    if kind == 0:
        return cv2.blur(img, (taille_kernel, taille_kernel), 0)
    if kind == 1:
        return cv2.medianBlur(img, taille_kernel)
    if kind == 3:
        return cv2.GaussianBlur(img, taille_kernel)
    # else return the current image
    return img


"""
Closing <- remove black noise inside white object
Opening <- remove white noise outside white object

kind = 0:closing; 1:opening; 2:substraction between dilatation and erosion
Default : closing
"""
def morphology(img, taille_kernel, kind=0):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (taille_kernel, taille_kernel))
    if kind == 0:
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if kind == 1:
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if kind == 2:
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    # else the current img
    return img


def draw_matches(img1, kp1, img2, kp2, matches):
    """
    Implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out = img2.copy()
    # Place the first image to the left
    # out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    # out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        # cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        # cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    return out


def isPerimeterOK(peri):
    peri_min = HEIGHT_MIN*2+WIDTH_MIN*2
    peri_max = HEIGHT_MAX*2+WIDTH_MAX*2
    if peri > peri_min or peri < peri_max:
        return False
    return True

def isAreaOK(area):
    area_min = HEIGHT_MIN*WIDTH_MIN
    area_max = HEIGHT_MAX*WIDTH_MAX
    if area > area_min or area < area_max:
        return False
    return True


def rectify(h):
    # order the box by top_left,top_right,bottom_right,bottom_left
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    return hnew

def order_corners(cnt):
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.cv.BoxPoints(rect))

    box = rectify(box)
    return box

def imgHomot(gray, cnt):
    box = order_corners(cnt)
    # getting the tag shape in pixel depending on its corners
    new_h, new_w = shape_contour(box) #(449,449)
    # TODO : change to :
    print new_h,new_w
    print "euh"
    # h = np.array([[0, 0], [new_h-1, 0], [0, new_w-1], [new_h-1,new_w-1]], np.float32)
    # according to http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#perspective-transformation
    h = np.array([[0, 0], [new_w-1, 0], [new_w-1,new_h-1], [0, new_h-1]], np.float32)
    retval = cv2.getPerspectiveTransform(box, h)
    warp = cv2.warpPerspective(gray, retval, (new_w,new_h))
    print warp.shape
    return warp


def canny_algorithm(img):
    """
    img : img matrix
    """

    # Blur
    # img = cv2.medianBlur(cv2.medianBlur(img, 3), 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Dilation/Erosion to close edges
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Canny edge detection using the computed median
    sigma = .10
    v = np.median(img)
    lower_thresh_val = int(max(0, (1.0 - sigma) * v))
    high_thresh_val = int(min(255, (1.0 + sigma) * v))
    # high_thresh_val, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lower_thresh_val = 0.5*high_thresh_val
    img = cv2.Canny(img, lower_thresh_val, high_thresh_val)
    return img

def canny_lecture(image):
    # Blur
    # img = cv2.medianBlur(cv2.medianBlur(img, 3), 3)
    at = time.time()
    img = cv2.GaussianBlur(image, (3, 3), 0)
    """
    bt = time.time()
    grad = gradientSobelXY(img)
    print 'temps gaussian : ',bt-at
    print 'temps gradient : ',time.time()-bt
    bt = time.time()
    lapl = cv2.Laplacian(img,cv2.CV_64F)
    print 'temps laplacian : ',time.time()-bt
    cv2.imshow("gaussizn lecture", img)
    bt = time.time()
    conv = convoluer(image)
    print 'temps conv : ',time.time()-bt
    cv2.imshow("init lecture", image)
    cv2.imshow("convolution lecture", conv)
    cv2.imshow("grad lecture", grad)
    cv2.imshow("laplacian lecture",lapl)
    dt = time.time()
    dil = cv2.dilate(grad, None, iterations = 1)
    print 'temps dilate : ',time.time()-dt
    cv2.imshow("grad dil lecture",dil)
    dt = time.time()
    erod1 = cv2.erode(dil, None, iterations = 1)
    print 'temps erode 1 : ',time.time()-dt
    cv2.imshow("grad dilerode 1 lecture",erod1)
    dt = time.time()
    thresh = thresholding(erod1,reverse=True,with_otsu=True)
    print 'temps thresh 1 : ',time.time()-dt
    cv2.imshow("thresh 1 lecture",thresh)

    dt = time.time()
    erod2 = cv2.erode(dil, None, iterations = 2)
    print 'temps erode 2 : ',time.time()-dt
    cv2.imshow("grad dilerode 2 lecture",erod2)
    dt = time.time()
    thresh = thresholding(erod2, reverse=True, with_otsu=True)
    print 'temps thresh 2 : ',time.time()-dt
    cv2.imshow("thresh 2 lecture", thresh)


    dt = time.time()
    erod2 = cv2.erode(dil, None, iterations = 3)
    print 'temps erode 3  : ',time.time()-dt
    cv2.imshow("grad dilerode 3 lecture",erod2)
    dt = time.time()
    thresh = thresholding(erod2,reverse=True,with_otsu=True)
    print 'temps thresh 3 : ',time.time()-dt
    cv2.imshow("thresh 3 lecture",thresh)
    # Dilation/Erosion to close edges
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Canny edge detection using the computed median
    
    # low, high=1.5*low
    dt = time.time()
    lower_thresh_val, thresh_im = cv2.threshold(img, THRESH_VALUE(), 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    high_thresh_val = 1.5 * lower_thresh_val
    canny = cv2.Canny(img, lower_thresh_val, high_thresh_val)
    print 'temps canny 1.5 : ',time.time()-dt
    cv2.imshow("canny lecture 1.5",canny)
    # high, low = 0.3*high
    dt = time.time()
    high_thresh_val, thresh_im = cv2.threshold(img, THRESH_VALUE(), 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    lower_thresh_val = 0.3 * high_thresh_val
    canny = cv2.Canny(img, lower_thresh_val, high_thresh_val)
    print 'temps canny 0.3 : ',time.time()-dt
    cv2.imshow("canny lecture 0.3",canny)
    # high, low = 0.7*high
    dt = time.time()
    high_thresh_val, thresh_im = cv2.threshold(img, THRESH_VALUE(), 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    lower_thresh_val = 0.7 * high_thresh_val
    canny = cv2.Canny(img, lower_thresh_val, high_thresh_val)
    print 'temps canny 0.7 : ',time.time()-dt
    cv2.imshow("canny lecture 0.7",canny)
    # low, high=1*low
    dt = time.time()
    lower_thresh_val, thresh_im = cv2.threshold(img, THRESH_VALUE(), 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    high_thresh_val = 1 * lower_thresh_val
    canny = cv2.Canny(img, lower_thresh_val, high_thresh_val)
    print 'temps canny 1 : ',time.time()-dt
    cv2.imshow("canny lecture 1",canny)
    """
    # high, low = 0.5*high
    dt = time.time()
    high_thresh_val, thresh_im = cv2.threshold(img, THRESH_VALUE(), 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    lower_thresh_val = 0.3 * high_thresh_val
    canny = cv2.Canny(img, lower_thresh_val, high_thresh_val)
    #print 'temps canny 0.5 : ',time.time()-dt
    #cv2.imshow("canny lecture 0.5",canny)

    #cv2.waitKey(0)
    # end
    return canny


def canny_algorithm_v2(image):
    """
    img : img matrix
    """

    # Blur
    # img = cv2.medianBlur(cv2.medianBlur(img, 3), 3)
    img = cv2.GaussianBlur(image, (3, 3), 0)

    # Dilation/Erosion to close edges
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Canny edge detection using the computed median
    high_thresh_val, thresh_im = cv2.threshold(img, THRESH_VALUE(), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    """
    per = CANNY_PERCENTAGE
    if per<1 :
	lower_thresh_val = per * high_thresh_val
	imge = cv2.Canny(img, lower_thresh_val, high_thresh_val)
    else :
    	min_thresh_val = high_thresh_val
	max_thresh_val = per*high_thresh_val
    	imge = cv2.Canny(img, min_thresh_val, max_thresh_val)
    if CANNY_VIDEO_MAKER.has_key(per):
        out = CANNY_VIDEO_MAKER[per]
    else:
        # define the codec and create videowriter object
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(VIDEO_PATH+"canny_"+str(per)+"param.avi",fourcc,20.0,(SIZE_X,SIZE_Y))
        CANNY_VIDEO_MAKER[per] = out
    # write the image
    out.write(imge)
    cv2.imshow("zze",imge)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
    	for out in CANNY_VIDEO_MAKER.values():
    	    out.release()
    """
    lower_thresh_val = CANNY_PERCENTAGE * high_thresh_val
    img = cv2.Canny(img, lower_thresh_val, high_thresh_val)
    # end
    return img


def our_canny_algorithm(img):
    # find regions of the image that have high horizontal gradients and low vertical gradients
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gray = cv2.equalizeHist(img)
    # cv2.imshow("equ",gray)
    (_, gr1) = cv2.threshold(gray, THRESH_VALUE(), 255, cv2.THRESH_BINARY_INV)# + cv2.THRESH_OTSU)
    # cv2.imshow("threshed",gr1)
    ### apply Sobel gradient
    gradX = cv2.Sobel(gr1, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gr1, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient1 = cv2.subtract(gradX, gradY)
    gradient1 = cv2.convertScaleAbs(gradient1)
    return gradient1

def check_tags(image,gray, tagz_cont,demo=False):
    """
    Foreach tag's contours found in the image, checking if it's a good one
    Return the list of tuples with (robot's number, robot's orientation, robot's coordinates)
    """
    views = list()
    # foreach contour found
    for cnt in tagz_cont:
        # take the contour and compute its bounding box
        # computing the min area bounding/fitting rect (even rotated)
        # getting the translation of the tag
        tag = imgHomot(gray, cnt)
        # check the id, orientation and coordinates of the bot
        # reading the info in the tag
        data =  tagid.lecture_tag(gray, tag, cnt)
        print "DATA =",data
        if data is None:
            # wrong tags in red
            cv2.drawContours(image,[cnt],-1,(0,0,255),2)
            continue
        if demo: # tag found in green
            cv2.drawContours(image,[cnt],-1,(0,255,0),2)
            cv2.imshow("tag found",tag)
        # adding them to the list views
        views.append(data)
    return views

def verify_hierarchy(hierarchy, i):
    #next = hierarchy[0, i, 0]
    #prev = hierarchy[0, i, 1]
    son = hierarchy[i, 2]
    #dad = hierarchy[0, i, 3]
    if son==-1:
        return False
    enough_sons = 0
    for nc,c in enumerate(hierarchy):
        if nc==i: continue
        # if the hierarchy got i a a parent
        if c[3]==i:
            enough_sons+=1
    #print i," got ",enough_sons," children"
    return (enough_sons>=4) # 5-1 ar securite

def apply_filters(img):
    # Faster without np
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img3 = cdf[img]
    tt = time.time()
    #ret, thresh = cv2.threshold(img3, THRESH_VALUE(), 255, cv2.THRESH_BINARY)
    ct = time.time()
    ret, thresh = cv2.threshold(img3, THRESH_VALUE(), 255, cv2.THRESH_BINARY)
    gt = time.time()
    #ret, thresh3 = cv2.threshold(img3, 135, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ft = time.time()
    """
    print "temps 2.04*brightness :",ct-tt
    print "temps hardcode :",gt-ct
    print "temps otsu :",ft-gt
    TPS_OTSU.append(ft-gt)
    TPS_CALC.append(ct-tt)
    TPS_FIX.append(gt-ct)
    cv2.imshow("tag threshed calc",thresh)
    cv2.imshow("tag threshed hard",thresh2)
    cv2.imshow("tag threshed auto",thresh3)
    if (cv2.waitKey(0) & 0xFF) == ord('s'):
        cv2.imwrite(IMG_PATH+"tag_threshed_calc.png", thresh)
        cv2.imwrite(IMG_PATH+"tag_threshed_hard.png", thresh2)
        cv2.imwrite(IMG_PATH+"tag_threshed_otsu.png", thresh3)
    """
    return thresh

def shape_contour(contour):
    """
    Only for contour with specific format top left,top right, bottom
    """
    width = max(contour[1][0]-contour[0][0], contour[3][0]-contour[2][0])
    height = max(contour[3][1]-contour[0][1],contour[2][1]-contour[1][1])
    return height,width
	
def verifChild(hierarchy, contour,peri_dad,area_dad):
    epsilon = 0.06*cv2.arcLength(contour, True) # aproximation accuracy
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx)!=4 or not cv2.isContourConvex(approx):
        # not what we are searching for
        return 0,None
    # veifying the height and width
    approx = rectify(approx) # getting top left,top right, bottom left, bottom right
    print "faisable"
    height, width = shape_contour(approx)
    peri = (height+width)*2
    area = height*width
    peri_per = 100*float(peri)/peri_dad
    area_per = 100*float(area)/area_dad
    print "peri = ",peri_per,"%"
    print "area = ",area_per,"%"
    # the perimeter must be ~50% and area ~23% to be the direction child
    if peri_per<=60 and peri_per>=35 and area_per<=35 and area_per>=10:
        print "un fils direction"
        return 2,approx
    # the perimeter must be ~70% and area ~50% to be the id child
    if peri_per<=85 and peri_per>=55 and area_per<=70 and area_per>=35:
        print "un fils id"
        return 1,approx
    # otherwise : not an interessant child
    return 0,None
