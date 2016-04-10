import cv2
import numpy as np

tag_in_scene = cv2.imread("tag3.jpg",0)
tag = cv2.imread("tag.png",0)
tag_in_scene = cv2.resize(tag_in_scene,(300,500))

def useSIFT(img):
    pass

def useSURF(img_init,img_in_scene):
    surf = cv2.SURF(400)
    # TODO a remove
    surf.hessianThreshold = 50000
    surf.upright = True
    kp,des = surf.detectAndCompute(img_in_scene,None)
    img2 = cv2.drawKeypoints(img_in_scene,kp,None,(0,0,255),4)
    return img2

# ORB
orb = cv2.ORB()
# SIFT
kp1,des1 = orb.detectAndCompute(tag,None)
kp2,des2 = orb.detectAndCompute(tag_in_scene,None)
# BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = bf.match(des1,des2)

#img3 = cv2.drawMatches(tag,kp1,tag_in_scene,kp2,matches[:10],flags=2)
img3 = tag_in_scene.copy()

def matchingTemplate(img_init,img_in_scene):
    # match template
    w,h = tag.shape[::-1]
    img_test = img_in_scene.copy()
    res = cv2.matchTemplate(img_test,img_init,cv2.TM_CCOEFF_NORMED)
    minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(res)
    top_left = maxLoc
    bottom_right = (top_left[0]+w,top_left[1]+h)
    cv2.rectangle(img_test,top_left,bottom_right,255,2)
    return img_test

# SURF
img3 = matchingTemplate(tag,tag_in_scene)


#cv2.imshow("tag in scene",tag_in_scene)
#cv2.imshow("tag a trouver",tag)
cv2.imshow("tests",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()