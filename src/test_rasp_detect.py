# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

"""

# import the necessary packages
import numpy as np
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time, datetime
# from our files
import tools
import found_tag_box as tg
import gestIO as io

### FUNCTIONS DEFINITIONS
def testImgFile(filename):
    camera, rawCapture = initCam()
    
    camera.capture(rawCapture, format="gray", use_video_port=True)
    frame = rawCapture.array
    # Display the resulting frame
    cv2.imshow("avant",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# working? :
def initCam():
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (tools.SIZE_X, tools.SIZE_Y)
    #camera.framerate = 64
    rawCapture = PiRGBArray(camera, size=(tools.SIZE_X, tools.SIZE_Y))
    # allow the camera to warmup
    time.sleep(0.1)
    return camera, rawCapture
    
# best way to take pictures
def takeVideo2():
    with PiCamera() as camera:
        camera.start_preview()
        time.sleep(2)
        with PiRGBArray(camera) as stream:
            camera.capture(stream, format='bgr')
            # At this point the image is available as stream.array
            image = stream.array
            print image


def takeVideo():
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mmin%Ssec')
    string="\n Prises du {} (avec find contours)\n".format(st)
    string+="***Comparaison des filtres sur algos***"
    i = 0
    camera,rawCapture = initCam()
    nbImgSec = 0
    tag = None
    print "Starting video"
    dt = st = time.time()
    #camera.start_preview()
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        nbImgSec += 1
        i += 1
        string += "\n--------> Prise n°{}".format(i)
        # time.sleep(0.1)

        # tests sur l'image
        # tps = tg.found_tag_img_comp(image)
        dt = time.time()
        results = tg.found_tag_img(image)
        tps = "\nTemps = "+str(time.time() - dt)
        # writing if there was or not any tag in the image
        if results==[]:
            tps+= " ---> No tag found"
        else:
            tps += " ---> Tags found:"+str(results)
        print "Résultats = ",results
        time.sleep(0.5)
        string += str(tps)
        if tag is None:
            # continue
            tag = image.copy()
        ft = time.time()
        # print 'Temps mis = {}'.format(ft-dt)
        # show the frame
        # cv2.imshow("origin", image)
        # cv2.imshow("fin", tag)
        key = cv2.waitKey(1) & 0xFF

        dt = time.time()
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or i>=tools.ITERATIONS:
            break
        if(dt-st>=1):
            string+= "\n\t1 seconde écoulée : {} images prises".format(nbImgSec)
            st=ft
            nbImgSec = 0
            
    # end with
    print "Fin prise"
    # When everything done, release the capture
    #camera.stop_preview()
    cv2.destroyAllWindows()
    io.writeOutputFile(string)

def applyFiltre(img):
    nimg = cv2.equalizeHist(img)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    #cv2.imwrite(tools.IMG_PATH+st+".png",nimg)
    return nimg
    
       
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
    min_size = thresh.size/20
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
    
    cv2.imshow("ma lect",thresh)
    #return (identity,orientation)
