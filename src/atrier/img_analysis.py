import cv2
import numpy as np
from matplotlib import pyplot as plt
import constants as cst
import time

tagName = cst.IMG_PATH+"tag_view.png"

img = cv2.imread(tagName,0)

dt = time.time()
width, height = img.shape
height -= 1
width -= 1
supp = cst.SUPP_MARGIN
margin = cst.OCCLUSION_MARGIN*width/cst.TAG_WIDTH
# computing the edges percentages
col_per = margin/width
lign_per = margin/height
# retrieving the rightmost border
col1 = img[0+supp:width-supp,0+supp:height*col_per-supp]
# retrieving the leftmost border
col2 = img[0+supp:width-supp,height-height*col_per+supp:height-supp]
# retrieving the upper border
lign1 = img[0+supp:width*lign_per-supp,0+supp:height-supp]
# retrieving the lower border
lign2 = img[width-width*lign_per+supp:width-supp,0+supp:height-supp]

hist = cv2.calcHist([lign1],[0],None,[256],[0,256])
plt.plot(hist)
plt.figure()
hist2 = cv2.calcHist([lign2],[0],None,[256],[0,256])
plt.plot(hist2)
plt.figure()
hist3 = cv2.calcHist([col1],[0],None,[256],[0,256])
plt.plot(hist3)
plt.figure()
hist4 = cv2.calcHist([col2],[0],None,[256],[0,256])
plt.plot(hist4)
plt.show()

"""
per1 = float(np.count_nonzero(lign1))/(len(lign1)*len(lign1[0]))
if not (per1>0.33 and per1<0.55):
    print False,"per1",per1
    exit()
per2 = float(np.count_nonzero(lign2))/(len(lign2)*len(lign2[0]))
if not (per2>0.33 and per2<0.55):
    print False,"per2",per2
    exit()
per3 = float(np.count_nonzero(col1))/(len(col1)*len(col1[0]))
if not (per3>0.33 and per3<0.55):
    print False,"per3",per3
    exit()
per4 = float(np.count_nonzero(col2))/(len(col2)*len(col2[0]))
if not (per4>0.33 and per4<0.55):
    print False,"per4",per4
    exit()
print True
print per1,per2,per3,per4
# time testing
"""
ft = time.time()
print "Ca prend ",ft-dt

# showing for tests
cv2.imshow("init",img)
cv2.imshow("lign1",lign1)
cv2.imshow("lign2",lign2)
cv2.imshow("col1",col1)
cv2.imshow("col2",col2)
#cv2.waitKey(0)
