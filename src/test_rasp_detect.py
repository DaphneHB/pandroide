# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

"""


# import the necessary packages
import numpy as np
import cv2,os
from picamera import PiCamera
from picamera.array import PiRGBArray
import time, datetime
# from our files
import tools
import found_tag_box as tg
import gestIO as io

def initCam():
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (tools.SIZE_X, tools.SIZE_Y)
    #camera.framerate = 64
    rawCapture = PiRGBArray(camera, size=(tools.SIZE_X, tools.SIZE_Y))
    # allow the camera to warmup
    time.sleep(0.1)
    return camera, rawCapture

### FUNCTIONS DEFINITIONS
def testImgFile(filename):
    camera, raw_capt = initCam()

    camera.capture(raw_capt, format="bgr", use_video_port=True)
    frame = raw_capt.array
    # Display the resulting frame
    cv2.imshow("avant",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def take_images():
    i = 0
    camera, rawCapture = initCam()
    print "Starting video"
    #camera.start_preview()
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        # Capture frame-by-frame
        i += 1
        cv2.imshow("img", image)
        cv2.imwrite(tools.IMG_PATH+"tag_view"+str(i)+".png",image)
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or i >= 1000:
            break
    # end with
    print "\nFin prise"
    # When everything done, release the capture
    # camera.stop_preview()
    cv2.destroyAllWindows()


def test_images():
    ststr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mmin%Ssec')
    string="\n Prises du {} \n".format(ststr)
    string+="***Calculs des true/false-positives/negatives***"
    print "\nDébut tests"
    nbImgSec = 0
    st = time.time()
    accepted = [(3,"10"),(12,"10"),(10,"00"),(0,"00")]
    list_img = sorted(os.listdir(tools.IMG_PATH))
    for i,img_name in enumerate(list_img):
        img = cv2.imread(tools.IMG_PATH+img_name)
        nbImgSec += 1
        string += "\n--------> Prise n°{}".format(i)

        dt = time.time()
        results = tg.found_tag_img(img)
        ft = time.time()
        tps = "\nTemps = " + str(ft - dt)
        if results == []:
            tps += " ---> No tag found"
        else:
            tps += " ---> Tags found:" + str(results)
        #print "Résultats image",i," = ", results
        string += str(tps)
        if results == []:
            tools.FALSE_NEG+=1
        else:
            if not any([x in accepted for x in results]):
                # there is no good one
                tools.FALSE_NEG+=1
            else:
                tools.TRUE_POS+=1
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
        if (dt - st >= 1):
            string += "\n\t1 seconde écoulée : {} images prises".format(nbImgSec)
            st = time.time()
            nbImgSec = 0

    #print list_img
    cv2.destroyAllWindows()
    io.writeOutputFile(string)

    print "\nFin tests"


def takeVideo():
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mmin%Ssec')
    string="\n Prises du {} (avec find contours) sur {} itérations\n".format(st,tools.ITERATIONS)
    string+="***Comparaison des filtres sur algos***"
    i = 0
    nbImgSec = 0
    tag = None
    camera, rawCapture = initCam()
    print "Starting video"
    # camera.start_preview()
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
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
        #time.sleep(0.5)
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
        rawCapture.truncate(0)

        dt = time.time()
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or i>=tools.ITERATIONS:
            break
        if(dt-st>=1):
            string+= "\n\t1 seconde écoulée : {} images prises".format(nbImgSec)
            st=ft
            nbImgSec = 0

    # end with
    print "\nFin prise"
    # When everything done, release the capture
    #camera.stop_preview()
    cv2.destroyAllWindows()
    io.writeOutputFile(string)