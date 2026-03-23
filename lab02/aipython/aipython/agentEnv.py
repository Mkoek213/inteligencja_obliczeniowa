# agentEnv.py - Agent environment
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import math
from agents import Environment
import matplotlib.pyplot as plt
import time

class Rob_body(Environment):
    def __init__(self, world, init_pos=(0,0), init_dir=90):
        """ world is the current world
        init_pos is a pair of (x-position, y-position)
        init_dir is a  direction in degrees; 0 is to right, 90 is straight-up, etc
        """
        self.world = world
        self.rob_pos = init_pos
        self.rob_dir = init_dir
        self.turning_angle = 18   # degrees that a left makes
        self.whisker_length = 6   # length of the whisker
        self.whisker_angle = 30   # angle of whisker relative to robot
        self.crashed = False

    def percept(self):
        return {'rob_pos':self.rob_pos,
                'rob_dir':self.rob_dir, 'whisker':self.whisker(), 'crashed':self.crashed}
    initial_percept = percept  # use percept function for initial percept too

    def do(self, action):
        """ action is {'steer':direction}
        direction is 'left', 'right' or 'straight'.
        Returns current percept.
        """
        if self.crashed:
            return self.percept()
        direction = action['steer']  
        compass_deriv = {'left':1,'straight':0,'right':-1}[direction]*self.turning_angle
        self.rob_dir = (self.rob_dir + compass_deriv +360)%360  # make in range [0,360)
        x,y = self.rob_pos
        rob_pos_new = (x + math.cos(self.rob_dir*math.pi/180), 
                       y + math.sin(self.rob_dir*math.pi/180))
        path = (self.rob_pos,rob_pos_new)
        if any(line_segments_intersect(path,wall) for wall in self.world.walls):
            self.crashed = True
        self.rob_pos = rob_pos_new
        self.world.do({'rob_pos':self.rob_pos,
                           'crashed':self.crashed, 'whisker':self.whisker()})
        return self.percept()

    def whisker(self):
        """returns true whenever the whisker sensor intersects with a wall
        """
        whisk_ang_world = (self.rob_dir-self.whisker_angle)*math.pi/180
            # angle in radians in world coordinates
        (x,y) = self.rob_pos
        wend = (x + self.whisker_length * math.cos(whisk_ang_world),
                y + self.whisker_length * math.sin(whisk_ang_world))
        whisker_line = (self.rob_pos, wend)
        hit = any(line_segments_intersect(whisker_line,wall)
                    for wall in self.world.walls)
        return hit
    
def line_segments_intersect(linea, lineb):
    """returns true if the line segments, linea and lineb intersect.
    A line segment is represented as a pair of points.
    A point is represented as a (x,y) pair.
    """
    ((x0a,y0a),(x1a,y1a)) = linea
    ((x0b,y0b),(x1b,y1b)) = lineb
    da, db = x1a-x0a, x1b-x0b
    ea, eb = y1a-y0a, y1b-y0b
    denom = db*ea-eb*da
    if denom==0:    # line segments are parallel
        return False
    cb = (da*(y0b-y0a)-ea*(x0b-x0a))/denom  # intersect along line b
    if cb<0 or cb>1:
        return False   # intersect is outside line segment b
    ca = (db*(y0b-y0a)-eb*(x0b-x0a))/denom # intersect along line a
    return 0<=ca<=1  # intersect is inside both line segments

# Test cases:
# assert line_segments_intersect(((0,0),(1,1)),((1,0),(0,1)))
# assert not line_segments_intersect(((0,0),(1,1)),((1,0),(0.6,0.4)))
# assert line_segments_intersect(((0,0),(1,1)),((1,0),(0.4,0.6)))

import math
from display import Displayable 
import matplotlib.pyplot as plt

class World(Environment):
    def __init__(self, walls = {}, locations = {}, plot_size=(-10,120,-10,60)):
        """walls is a set of line segments 
               where each line segment is of the form ((x0,y0),(x1,y1))
         locations is a loc:pos dictionary 
            where loc is a named location, and pos is an (x,y) position.
        """
        self.walls = walls
        self.locations = locations
        self.loc2text = {}
        self.history = [] # list of (pos, whisker, crashed)
        # The following control how it is plotted
        plt.ion()
        fig, self.ax = plt.subplots()
        #self.ax.set_aspect('equal')
        self.ax.axis(plot_size)
        self.sleep_time = 0.05     # time between actions (for real-time plotting)
        self.draw()

    def do(self, action):
        """action is {'rob_pos':(x,y), 'whisker':Boolean, 'crashed':Boolean}
        """
        self.history.append((action['rob_pos'],action['whisker'],action['crashed']))
        x,y = action['rob_pos']
        if action['crashed']:
            self.display(1, "*Crashed*")
            self.ax.plot([x],[y],"r*",markersize=20.0)
        elif action['whisker']:
            self.ax.plot([x],[y],"ro")
        else:
            self.ax.plot([x],[y],"go")
        plt.draw()
        plt.pause(self.sleep_time)
        return {'walls':self.walls}
   
    def draw(self):
        for wall in self.walls:
            ((x0,y0),(x1,y1)) = wall
            self.ax.plot([x0,x1],[y0,y1],"-k",linewidth=3)
        for loc in self.locations:
            self.plot_loc(loc)

    def plot_loc(self, loc):
        (x,y) = self.locations[loc]
        if loc in self.loc2text:
            for e in  self.loc2text[loc]:
                e.remove()  # e.set_visible(False)
        self.loc2text[loc] = ( self.ax.text(x,y,"*",ha="center",va="center",size=20),
                         self.ax.text(x+2.0,y+1,loc)) # label above and to the right

