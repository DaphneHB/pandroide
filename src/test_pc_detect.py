# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

"""

# import the necessary packages
import cv2
import time, datetime
# from our files
import tools
import found_tag_box as tg
import gestIO as io

### FUNCTIONS DEFINITIONS
def testImgFile(filename):
    camera = cv2.VideoCapture(0)

    frame = camera.read()
    # Display the resulting frame
    cv2.imshow("avant",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def take_images():
    i = 0
    camera = cv2.VideoCapture(0)
    print "Starting video"
    while (True):
        time.sleep(0.1)
        # Capture frame-by-frame
        _, image = camera.read()
        i += 1
        cv2.imshow("img", image)
        cv2.imwrite(tools.IMG_PATH+"tag_view"+str(i)+".png",image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q") or i >= tools.ITERATIONS:
            break
    # end with
    print "\nFin prise"
    # When everything done, release the capture
    # camera.stop_preview()
    cv2.destroyAllWindows()


def takeVideo():
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mmin%Ssec')
    string="\n Prises du {} (avec find contours) sur {} itérations\n".format(st,tools.ITERATIONS)
    string+="***Comparaison des filtres sur algos***"
    i = 0
    camera = cv2.VideoCapture(0)
    nbImgSec = 0
    tag = None
    print "Starting video"
    dt = st = time.time()
    #camera.start_preview()
    # capture frames from the camera
    while (True):
        # time.sleep(0.1)
        # Capture frame-by-frame
        _, image = camera.read()
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
