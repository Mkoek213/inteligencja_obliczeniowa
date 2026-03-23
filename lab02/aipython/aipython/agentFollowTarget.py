# agentFollowTarget.py - Plotting for moving targets
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import matplotlib.pyplot as plt
from agentEnv import Rob_body, World
from agentMiddle import Rob_middle_layer
from agentTop import Rob_top_layer

class World_follow(World):
    def __init__(self, walls = {}, locations = {}, epsilon=5):
        """plot the agent in the environment. 
        epsilon is the threshold how how close someone needs to click to select a location.
        """
        self.epsilon = epsilon
        World.__init__(self, walls, locations)
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.pressloc = None
        for loc in self.locations:
            self.display(2,f"    loc {loc} at {self.locations[loc]}")
            
    def on_press(self, event):
        print("press", event)
        self.display(2,'v',end="")
        self.display(2,f"Press at ({event.xdata},{event.ydata}")
        self.pressloc = None
        if event.xdata: 
          for loc in self.locations:
            lx,ly = self.locations[loc]
            if abs(event.xdata- lx) <= self.epsilon and abs(event.ydata- ly) <= self.epsilon :
                self.display(2,f"moving {loc} from ({event.xdata}, {event.ydata})" )
                self.pressloc = loc

    def on_release(self, event):
        self.display(2,'^',end="")
        if self.pressloc is not None and event.xdata:
            self.display(2,f"Placing {self.pressloc} at {(event.xdata, event.ydata)}")
            self.locations[self.pressloc] = (event.xdata, event.ydata)
            self.plot_loc(self.pressloc)
        self.pressloc = None

    def on_move(self, event):
        if self.pressloc is not None and event.inaxes: 
            self.display(2,'-',end="")
            self.locations[self.pressloc] = (event.xdata, event.ydata)
            self.plot_loc(self.pressloc)
        else:
            self.display(2,'.',end="")

def rob_follow():
    global world, body, middle, top
    world = World_follow(walls = {((20,0),(30,20)), ((70,-5),(70,25))},
                  locations = {'mail':(-5,10), 'o103':(50,10),
                                   'o109':(100,10),'storage':(101,51)})
    body = Rob_body(world)
    middle = Rob_middle_layer(body)
    top = Rob_top_layer(middle, world)

# top.do({'visit':['o109','storage','o109','o103']})

if __name__ == "__main__":
    rob_follow()
    print("Try: top.do({'visit':['o109','storage','o109','o103']})")

