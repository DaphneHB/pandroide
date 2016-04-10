import cv2
import numpy as np

# Normal routines
img = cv2.imread('tag1.jpg')
img = cv2.resize(img,(700,700))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,50,255,1)

# Remove some small noise if any.
dilate = cv2.dilate(thresh,None)
erode = cv2.erode(dilate,None)

# Find contours with cv2.RETR_CCOMP
contours,hierarchy = cv2.findContours(erode,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)            
# loop over our contours
"""for cnt in contours:
	# approximate the contour
	peri = cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
 
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
"""
for i,cnt in enumerate(contours):
    # approximate the contour
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    # Check 
    if len(approx)==4 and hierarchy[0,i,2]!=-1:
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)
       
cv2.imshow('img',img)
cv2.imwrite('sofsqure.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()