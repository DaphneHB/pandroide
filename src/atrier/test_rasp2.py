import cv2
import numpy as np
import picamera
import track_tag as tck

# the name of the test img in the current directory
FNAME = 'tag2.jpg'
TAG_TEMP = cv2.imread('tag.png',0)

# treating an image with filters
# threshold <- only 2 colors (black/white)
# erosion <- marking the skin borders
def imgTreating(img):
	kernel = np.ones((5,5),np.uint8)
	img2 = tck.matchingTemplate(TAG_TEMP,img)
	# brightness equalization
	equ = cv2.equalizeHist(img2)
	# getting the black-white image
	ret,thresh = cv2.threshold(equ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# thicken the borders
	erosion = cv2.erode(thresh,kernel,iterations=1)
	# narrow the borders
	dilate = cv2.dilate(erosion,kernel,iterations=1)
	# 
	return thresh


# getting an image from the current directory
def fileImgTest(filename):
	img = cv2.imread(filename,0)
	img2 = cv2.resize(img,(700,700))
	# visualization of the image
	cv2.imshow('img',img2)
	cv2.imshow('traitee',imgTreating(img2))
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# taking pictures with the camera of the RaspberryPi
# applying the treatment picture by picture
def takeVideo():
	camera = picamera.PiCamera()
	while(True):
		# capturing a video
		camera.capture('image.jpg')
		# testing if the image match
		fileImgTest('image.jpg')
		# to quit the video capturing
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
### TESTS
#fileImgTest(FNAME)

takeVideo()
