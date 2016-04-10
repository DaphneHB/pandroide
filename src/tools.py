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


def houghlines(img):
    hough = img.copy()
    lines = cv2.HoughLines(img, 1, np.pi / 180, 200)
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


def threshold_baw(img, seuil, reverse=False, with_otsu=False, adaptative=False):
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


def match_features(img1, img2):
    min_match_count = 10
    ### TODO : best between SIFT & SURF &....
    # Initiate SIFT detector
    sift = cv2.SURF()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    ### TODO : best between FLANN & BF
    """ # FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    """

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        print "DST : {}".format(dst)
        cv2.polylines(img2, [np.int32(dst)], True, 255, 3)

    else:
        print "Not enough matches are found - %d/%d" % (len(good), min_match_count)
        # matchesMask = None
        return None
    # img3 = drawMatches(img1,kp1,img2,kp2,good)#,None,**draw_params)

    return img2


def gradientSobelXY(img):
    gradX = cv2.Sobel(img, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(img, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    return gradient


def dilatation(img, it):
    return cv2.dilate(img, None, iterations=it)


def erosion(img, it):
    return cv2.erode(img, None, iterations=it)


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
    # peri10 = 290*2+410*2
    peri20 = 2 * 190 + 2 * 216
    # peri30 = 110*2+165*2
    peri50 = 2 * 75 + 2 * 90
    if peri > peri20 or peri < peri50:
        return False
    return True


def isAreaOK(area):
    # area10 = 290*410
    area20 = 190 * 216
    # area30 = 110*165
    area50 = 75 * 90
    if area > area20 or area < area50:
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


def sort_corners(corners):
    # order the box by top_left,top_right,bottom_left,bottom_right
    top_corners = sorted(corners, key=lambda x: x[1])
    top = top_corners[:2]
    bot = top_corners[2:]
    if len(top) == 2 and len(bot) == 2:
        tl = top[1] if top[0][0] > top[1][0] else top[0]
        tr = top[0] if top[0][0] > top[1][0] else top[1]
        br = bot[1] if bot[0][0] > bot[1][0] else bot[0]
        bl = bot[0] if bot[0][0] > bot[1][0] else bot[1]
        corners = np.float32([tl, tr, br, bl])
    print "CORNERS SORTED = {}".format(corners)
    return corners

def curve_to_quadrangle(points):
    assert points.size == 8, 'not a quadrangle'
    vertices = [p[0] for p in points]
    return np.float32([x for x in vertices])


def homothetie_marker(img_orig, corners):
    """
    Find the perspective transfomation to get a rectangular 2D marker
    """

    ideal_corners = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
    M = cv2.getPerspectiveTransform(corners, ideal_corners)
    marker2D_img = cv2.warpPerspective(img_orig, M, (200, 200))
    return marker2D_img


def imgHomot(gray, box):
    ### TESTS de Elyas
    """
    corners = curve_to_quadrangle(box)
    sorted_corners = sort_corners(corners)
    
    warp = homothetie_marker(gray,sorted_corners)
    #cv2.imwrite("warp.png",warp)
    """
    box = rectify(box)

    h = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)
    retval = cv2.getPerspectiveTransform(box, h)
    warp = cv2.warpPerspective(gray, retval, (450, 450))

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


def canny_algorithm_v2(img):
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
    high_thresh_val, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower_thresh_val = 0.5 * high_thresh_val
    img = cv2.Canny(img, lower_thresh_val, high_thresh_val)
    return img


def our_canny_algorithm(img):
    # find regions of the image that have high horizontal gradients and low vertical gradients
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gray = cv2.equalizeHist(img)
    # cv2.imshow("equ",gray)
    (_, gr1) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("threshed",gr1)
    ### apply Sobel gradient
    gradX = cv2.Sobel(gr1, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gr1, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient1 = cv2.subtract(gradX, gradY)
    gradient1 = cv2.convertScaleAbs(gradient1)
    return gradient1


def is_real_tag(tag):
    """
    Checking is the read tag really is a tag (not a window)
    Return True if the tag is a good one
    """
    dt = time.time()
    width, height = tag.shape
    height -= 1
    width -= 1
    supp = SUPP_MARGIN
    margin = OCCLUSION_MARGIN * width / TAG_WIDTH
    # computing the edges percentages
    col_per = margin / width
    lign_per = margin / height
    # retrieving the rightmost border
    col1 = tag[0+supp:width-supp,0+supp:height*col_per-supp]
    # retrieving the leftmost border
    col2 = tag[0+supp:width-supp,height-height*col_per+supp:height-supp]
    # retrieving the upper border
    lign1 = tag[0+supp:width*lign_per-supp,0+supp:height-supp]
    # retrieving the lower border
    lign2 = tag[width-width*lign_per+supp:width-supp,0+supp:height-supp]
    cv2.imshow("lign1", lign1)
    cv2.imshow("lign2", lign2)
    cv2.imshow("col1", col1)
    cv2.imshow("col2", col2)
    #cv2.waitKey(0)

    per1 = float(np.count_nonzero(lign1)) / (len(lign1) * len(lign1[0]))
    if not (per1 > 0.25 and per1 < 0.55):
        print False, "per1", per1
        return False
    per2 = float(np.count_nonzero(lign2)) / (len(lign2) * len(lign2[0]))
    if not (per2 > 0.25 and per2 < 0.55):
        print False, "per2", per2
        return False
    per3 = float(np.count_nonzero(col1)) / (len(col1) * len(col1[0]))
    if not (per3 > 0.25 and per3 < 0.55):
        print False, "per3", per3
        return False
    per4 = float(np.count_nonzero(col2)) / (len(col2) * len(col2[0]))
    if not (per4 > 0.25 and per4 < 0.55):
        print False, "per4", per4
        return False
    print True
    print per1, per2, per3, per4

    # time testing
    ft = time.time()
    print "Ca prend ", ft - dt
    return True


def check_tags(gray, tagz_cont):
    """
    Foreach tag's contours found in the image, checking if it's a good one
    Return the list of tuples with (robot's number, robot's orientation, robot's coordinates)
    """
    global IEME_TAG
    views = list()
    # foreach contour found
    for cnt in tagz_cont:
        # take the contour and compute its bounding box 
        # computing the min area bounding/fitting rect (even rotated)
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.cv.BoxPoints(rect))

        # getting the translation of the tag
        tag = imgHomot(gray, box)
        # thresholding for a better use after
        thresh = threshold_baw(tag,170)
        cv2.imshow("found", tag)
        time.sleep(0.1)
        # the contour found isn't a tag
        if not is_real_tag(thresh):
            continue
        else:
            IEME_TAG += 1
            cv2.imwrite(IMG_PATH+"tag_view"+str(IEME_TAG)+".png",tag)
            # else :
            # check the id, orientation and coordinates of the bot
            # reading the info in the tag
            data =  tagid.lecture_tag(tag)
            if data is None:
                continue
            # else : add them to the list views
            print data
            #views.append(data)
    return views

def apply_filters(img):
    # Faster without np?
    # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img3 = cdf[img]
    ret, thresh = cv2.threshold(img3, 127, 255, 0)
    return thresh