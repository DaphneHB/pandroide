# -*- coding: utf-8 -*-
#!/usr/bin/env python

import dbus
import dbus.mainloop.glib
import gobject
import traceback
import sys
import time
import os
import picamera
import termios
import tty
from optparse import OptionParser
import threading
import time
import cv2
from picamera.array import PiRGBArray
import re

proxSensorsVal=[0,0,0,0,0]

getch = None

curFreq = 392
delta = 10

curLine = 0
sounds = []

class Getch:
  def __init__(self):
    pass

  def __call__(self):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return ch

def dbusReply():
    pass

def dbusError(e):
    print 'error %s'
    print str(e)

def Music():
    # global curFreq

    # char = getch()
    # if char.lower() == 'q' or char.lower() == 'x' :
    #   return False
    # elif char.lower() == 'z' :
    #   curFreq += delta
    #   print("z :" + str(curFreq))
    #   network.SendEventName("SetSound", [curFreq, 1], reply_handler=dbusReply, error_handler=dbusError)
    # elif char.lower() == 's' :
    #   curFreq -= delta
    #   print("s :" + str(curFreq))
    #   network.SendEventName("SetSound", [curFreq, 1], reply_handler=dbusReply, error_handler=dbusError)
    # elif char.lower() == 'r' :
    #   print("r :" + str(curFreq))
    #   network.SendEventName("SetSound", [curFreq, 1], reply_handler=dbusReply, error_handler=dbusError)

    """
    global curLine
    global sounds

    stop = False
    print(curLine)
    if curLine == 4 :
      stop = True

    if curLine < len(sounds) and stop == False :
      print(sounds[curLine][0])
      network.SendEventName("SetSound", [sounds[curLine][0], sounds[curLine][1]/float(1000)], reply_handler=dbusReply, error_handler=dbusError)
    else :
      network.SendEventName("SetSound", [0, -60], reply_handler=dbusReply, error_handler=dbusError)
      loop.quit()
      return False
    """
    network.SendEventName("SetSound", [400, 1], reply_handler=dbusReply, error_handler=dbusError)
    
    time.sleep(1)
    return True
 
def get_variables_reply(r):
    global proxSensorsVal
    proxSensorsVal=r
 
def get_variables_error(e):
    print 'error:'
    print str(e)
    loop.quit()
 
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--system", action="store_true", dest="system", default=False,help="use the system bus instead of the session bus")
 
    (options, args) = parser.parse_args()

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
 
    if options.system:
        bus = dbus.SystemBus()
    else:
        bus = dbus.SessionBus()
 
    #Create Aseba network 
    network = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
    network.LoadScripts("asebaCommands.aesl", reply_handler=dbusReply, error_handler=dbusError)
    """
    global getch
    getch = Getch()

    global curLine
    curLine = 0
    if os.path.isfile("script.sh") :
      with open("script.sh", 'r') as fileSounds :
        fileSounds = fileSounds.readlines()

        global sounds
        sounds = []
        for line in fileSounds :
          s = re.search(r"^beep -f (\d+\.\d+) -l (\d+\.\d+)$", line.rstrip('\n'))
          if s :
            sounds.append(tuple([float(s.group(1)), float(s.group(2))]))
    """
    # To avoid conflicts between gobject and python threads
    gobject.threads_init()
    dbus.mainloop.glib.threads_init()
 
    #print in the terminal the name of each Aseba NOde
    print network.GetNodesList()  
    #GObject loop
    #print 'starting loop'
    loop = gobject.MainLoop()
    #call the callback of Braitenberg algorithm
    handle = gobject.timeout_add (100, Music) #every 0.1 sec
    loop.run()
