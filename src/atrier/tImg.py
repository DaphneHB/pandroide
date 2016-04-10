import numpy as np
import cv2
import time

#differentes matrices de convolution
k_contraste = np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,-1,5,-1,0],[0,0,-1,0,0],[0,0,0,0,0]])
k_bords = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,1,-4,1,0],[0,0,1,0,0],[0,0,0,0,0]])
kernel_float32 = np.ones((5,5),np.float32)/25
kernel_uint8 = np.ones((5,5),np.uint8)

tag_seul = cv2.imread('tag.png',0)
### gere et traite les images videos webcam
def takeVideo():
    cap = cv2.VideoCapture(0)
    
    while(True):
        time.sleep(0.2)
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ### Traitement de l'image.
        dt = time.time()
        torb = matchFeatures(tag_seul,gray)
        ft = time.time()
        print "Temps mis : {}".format(ft-dt)
        if torb is None :
            continue
        # pour gerer la lumiere
        prems = useHisto(torb)
        # pour n'avoir que 2 couleurs (noir/blanc)
        prems = seuillage_noir_blanc(prems)
        # pour uniformiser
        autres = exeCanny(prems)
        # pour recuperer les contours
        test2 = findCont(autres)
        # Display the resulting frame
        cv2.imshow('frame0',test2)
        #cv2.imshow('frameGray',gray)
        cv2.imshow('frame1',torb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def orb_test(img):
    orb = cv2.ORB()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp,des = orb.compute(img,kp)
    ret = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
    return ret

# differents effets et filtres
def useHisto(img):
    equ = cv2.equalizeHist(img)
    return equ

def houghlines(img):
    hough = img.copy()
    lines = cv2.HoughLines(img,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv2.line(hough,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite("houghlinesImg.png",hough)
    return hough

def findCont(copy):
    contours, hierarchy = cv2.findContours(copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return copy

def exeCanny(img):
    # on marque les contours
    edges = cv2.Canny(img,100,200, L2gradient=True)
    return edges
        
# meilleur : seuillage+otsu+closingMorpho
def seuillage_noir_blanc(img):
    #ret3,img3 = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    ret3,img3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #ret4,img3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img3

def convoluer(img,mat):
    # equivalent de dst
    return cv2.filter2D(img,-1,mat)

def matchFeatures(img1,img2):
    MIN_MATCH_COUNT = 10
    ### TODO : best between SIFT & SURF &....
    # Initiate SIFT detector
    sift = cv2.SURF()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)


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
    matches = bf.knnMatch(des1,des2, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #matchesMask = mask.ravel().tolist()
    
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        print "DST : {}".format(dst)
        cv2.polylines(img2,[np.int32(dst)],True,255,3)
    
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        #matchesMask = None
        return None
    #img3 = drawMatches(img1,kp1,img2,kp2,good)#,None,**draw_params)
    
    return img2



def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
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

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out = img2.copy()
    # Place the first image to the left
    #out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    #out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        #cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        #cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    return out

    
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
   




takeVideo()
"""img = cv2.imread("tag.png",0)
nouvimg = exeCanny(img)
truc2 = houghlines(nouvimg)
truc = houghlines(img)
cv2.imshow('frame0',truc)
cv2.imshow('frame2',truc2)
cv2.imshow('frame',nouvimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""