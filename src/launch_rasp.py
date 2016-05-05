# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import datetime
import threading
import gobject
# from our files
import tools
import found_tag_box as tg
import gestIO as io
import generePlots as plt

class RaspbThread(threading.Thread):
	def __init__(self,thy_controller,demo=False):
		threading.Thread.__init__(self)
		self.initCam()
		self.expect =  []
		self.thymio = thy_controller
		self._stop = False
		self.demo = demo
		tools.DIST_ANGLE = True
		self.daemon = True
		self.__stop = threading.Event()

	def initCam(self):
		# initialize the camera and grab a reference to the raw camera capture
		self.camera = PiCamera()
		self.camera.resolution = (tools.SIZE_X, tools.SIZE_Y)
		#camera.framerate = 64
		self.camera.brightness = tools.INIT_BRIGHTNESS
		self.rawCapture = PiRGBArray(self.camera, size=(tools.SIZE_X, tools.SIZE_Y))
		print "Initializing camera"
		# allow the camera to warmup
		time.sleep(3)

	def expected(self,tagz):
		print "Will expect to find these tags :",tagz
		self.expect = tagz

	def run(self):
		print "Launching thymio"
		i = nbImgSec = 0
		st = dt = time.time()
		print "\nStarting vizualization!"
		for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
			if self.__stop.isSet():#self._stop:
				print "if entered"
				break
			# grab the raw NumPy array representing the image, then initialize the timestamp
			# and occupied/unoccupied text
			image = frame.array
			nbImgSec += 1
			i += 1
			if i%tools.BRIGHTNESS_CHECK!=0:
				verif = tools.verify_brightness(image)
			else:
				verif = tools.verify_brightness(image,go=True)
			# there were a modification
			if verif!=0:
				print "Brightness",self.camera.brightness
				self.camera.brightness += verif            
			# tests sur l'image
			results = []
			results = tg.found_tag_img(image, demo = self.demo)
			print "\nTemps = "+str(time.time() - dt)
			# writing if there was or not any tag in the image
			if results==[]:
				print " ---> No tag found"
			elif any([x in self.expect for x in results]):
				#self.thymio.found_good()
				print "GOOD -> Robot seen : ",results
			else:
				#self.thymio.found_wrong()
				print "WRONG -> Robot seen : ",results
			# show the frame
			key = 1#cv2.waitKey(1) & 0xFF
			self.rawCapture.truncate(0)
			# not working : cv2 not showing
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):# or i>=10:#tools.ITERATIONS:
				self._stop = True
				break
			dt = time.time()
			"""
			if(dt-st>=1):
			#print "\n1seconde ecoulee : {} images prises".format(nbImgSec)
			st=ft
			nbImgSec = 0
			"""
		# end with
		print "\nEnd vizualization"
		# When everything's done, release the capture
		#camera.stop_preview()
		cv2.destroyAllWindows()
		#self.thymio.postActions()

	def stop(self) :
		print "Setting stop"
		self.__stop.set()

	def isStopped(self) :
		return self.__stop.isSet()

	def stopping(self):
		self.stop()
		self._stop = True
		self.thymio.stopping()


class RaspbClass():
	def __init__(self,thy_controller,demo=False):
		self.expect =  []
		self.thymio = thy_controller
		self._stop = False
		self.demo = demo or tools.DEMO
		print "demo =",self.demo
		self.daemon = True
		# activating the calculation of the distance and the angle
		tools.DIST_ANGLE = False
		self.diff_tagz = False
		self.initCam()
		
	def initCam(self):
		# initialize the camera and grab a reference to the raw camera capture
		self.camera = PiCamera()
		self.camera.resolution = (tools.SIZE_X, tools.SIZE_Y)
		#camera.framerate = 64
		self.camera.brightness = tools.INIT_BRIGHTNESS
		self.rawCapture = PiRGBArray(self.camera, size=(tools.SIZE_X, tools.SIZE_Y))
		print "Initializing camera"
		# allow the camera to warmup
		time.sleep(3)

	def tag_expected(self,tagz):
		self.diff_tagz = True
		print "Will expect to find these tags :",tagz
		self.expected = tagz
	
	def bot_expected(self,bots):
		self.diff_tagz = False
		print "Will expect to find these robot :",bots
		self.expected = bots
		
	def set_demo(self,is_demo):
		self.demo = is_demo
		
	def add_expected(self,robot):
		self.expected.append(robot)
	
	def verify_results(self, results):
		"""
		Verifying the robots found in the image area
		if their id and direction match
		"""
		found_ok = 0
		mistakes = 0
		for bot in results:
			if not self.diff_tagz:
				info = bot[0]
			else:
				info = (bot[0],bot[1])
			if info in self.expected:
				found_ok += 1
			else :
				mistakes += 1
		return found_ok, mistakes
		
	def start(self):
		# if we wan to run a demonstration of the robot capacities
		self.run()
			
	def run(self):
		ststr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%Hh%Mmin')
		self.log = "\n Prise du {}\n".format(ststr)
		self.log += "\n***Récupération des données de l'arène***"
		self.log += "\n\tInitial brightness = {}".format(self.camera.brightness)
		self.log += "\n\tCamera asked framerate = {}".format(self.camera.framerate)
		self.log += "\n\tCamera resolution = {}".format(self.camera.resolution)
		self.log += "Will expect to find these tags : {}".format(self.expected)
		print "Raspberry in process"
		i = 0
		nbImgSec = 0
		st = dt = time.time()
		print "\nStarting vizualization!"
		for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
			# stopping any noise
			if self._stop:
				break
			# grab the raw NumPy array representing the image, then initialize the timestamp
			# and occupied/unoccupied text
			image = frame.array
			nbImgSec += 1
			i += 1
			if i%tools.BRIGHTNESS_CHECK!=0:
				verif = tools.verify_brightness(image)
			else:
				verif = tools.verify_brightness(image,go=True)
			# there were a modification
			if verif!=0:
				self.camera.brightness += verif            
				self.log += "\n****** Brightness changed : {} ******\n".format(self.camera.brightness)
			# tests sur l'image
			results = tg.found_tag_img(image, demo = self.demo)
			print "\nTemps = "+str(time.time() - dt)
			self.log += "\nTemps = {}".format(time.time() - dt)
			# writing if there was or not any tag in the image
			if results==[]:
				print "|---> No tag found"
				self.log += "\n|---> No tag found"
			else:
				found, mis  = self.verify_results(results)
				if found ==0 :
					#self.thymio.found_wrong()
					print "|---> WRONG (0 good) -> Robot seen : ",results
					self.log += "\n|---> WRONG (0 good) -> Robot seen : {}".format(results)
					self.thymio.found_wrong()
				elif mis==0:
					#self.thymio.found_good()
					print "|---> ALLGOOD (0 wrong) -> Robot seen : ",results
					self.log += "\n|---> GOOD -> Robot seen : {}".format(results)
					self.thymio.found_good()
				else:
					#self.thymio.found_good()
					print "|---> GOOD ({} good - {} wrong) -> Robot seen : {}".format(found,mis,results)
					self.log += "\n|---> GOOD ({} good - {} wrong) -> Robot seen : {}".format(found,mis,results)
					#self.thymio.turnLeft(mis*10)
			if self.demo:
				key = cv2.waitKey(1) & 0xFF
				# not working : cv2 not showing
				# if the `q` key was pressed, break from the loop
				if key == ord("q") :#or i>=tools.ITERATIONS:
					self._stop = True
					break
			self.rawCapture.truncate(0)
			dt = time.time()
			if(dt-st-0.1>=1):
				print "\n1 seconde ecoulee : {} images prises".format(nbImgSec)
				self.log += "\n1 seconde ecoulee : {} images prises".format(nbImgSec)
				st=dt
				nbImgSec = 0
		# end for
		"""
		print "\nEnd vizualization"
		# When everything's done, release the capture
		#camera.stop_preview()
		io.writeOutputFile(string)
		if self.demo:
			cv2.destroyAllWindows()
		"""
		
	def stopping(self):
		self._stop = True
		print "\nEnd vizualization"
		io.writeOutputFile(self.log,log=True)
		if self.demo:
			cv2.destroyAllWindows()	
		self.thymio.stopping()

