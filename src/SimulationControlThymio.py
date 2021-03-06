# -*- coding: utf-8 -*-
#!/usr/bin/env/python

import threading
#import Simulation
#import Params
from thymio_tools import *

class SimulationControlThymio(threading.Thread) :
	def __init__(self) :
#	def __init__(self, controller, mainLogger) :
		#Simulation.Simulation.__init__(self, controller, mainLogger)
		threading.Thread.__init__(self)
		self.daemon = True
		self.__stop = threading.Event()

	def run(self):
		self.preActions()

		while not self.__stop.isSet() :
			self.step()

		self.postActions()
		print "Thymio stopped"

	def preActions(self) :
		pass

	def postActions(self) :
		stop()
	
	def pause(self):
		stop()
		time.sleep(1)
		
	def backward(self):
		go_backward()
		time.sleep(0.5)
		
	def turnLeft(angle):
		turn_left(angle)
		
	def turnRight(angle):
		turn_right(angle)
		
	def step(self) :
		random_walk()
		#self.pause()
		
	def stopping(self):
		self.__stop__()
		#self.postActions()
	
	def __stop__(self) :
		self.__stop.set()

	def isStopped(self) :
		return self.__stop.isSet()

	def found_good(self):
		good_noise()

	def found_wrong(self):
		bad_noise()

class Launch(threading.Thread):
	def __init__(self, noise_fct):
		threading.Thread.__init__(self)
		# loop for sound
		# GObject loop
		self.loop = gobject.MainLoop()
		self.loop_context = self.loop.get_context()
		handle = gobject.timeout_add(100, noise_fct)  # every 0.1 sec

	def run(self):
		self.loop.run()

	def stop(self):
		self.loop.quit()
