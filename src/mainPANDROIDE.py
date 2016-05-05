# -*- coding: utf-8 -*-
from SimulationControlThymio import * # thread controlling the thymio
#from launch_rasp import RaspbThread as Raspb# threading version 
from launch_rasp import RaspbClass as Raspb # function version 
#import from_rasp as raspberry # without class or thread

if __name__ == '__main__':
	runTime = 50
	thymio_sim = SimulationControlThymio()
	raspberry = Raspb(thymio_sim)
#	raspberry.tag_expected([(256,"10"),(320,"10"),(0,"00")])
	raspberry.bot_expected([341,98,15])
	raspberry.set_demo(DEMO)
	try:
		print "Launching thymio"
		thymio_sim.start()
		print "Thymio in process"
		# the prog relies on the raspberry
		print 'starting loop'
		raspberry.start()
		#raspberry.run(thymio_sim, expected = [(256,"10"),(320,"10"),(0,"00")], demo = False)
		print "Press Ctrl+c to stop"
	except KeyboardInterrupt:
		print "Ctrl+c -> Stopping"
		raspberry.stopping() # will also stop the thymio thread
	thymio_sim.join()
	#raspberry.join() # if the raspberry is a thread
	
	"""
	main_loop = gobject.MainLoop()
	#call the callback of Braitenberg algorithm
	handle = gobject.timeout_add (100, main.step) #every 0.1 sec
	main_loop.run()
	main_loop.quit()
	"""
