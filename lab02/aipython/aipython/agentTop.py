# agentTop.py - Top Layer
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable 
from agentMiddle import Rob_middle_layer
from agents import Agent, Environment

class Rob_top_layer(Agent, Environment):
    def __init__(self, lower, world, timeout=200 ):
        """lower is the lower layer
        world is the world (which knows where the locations are)
        timeout is the number of steps the middle layer goes before giving up
        """
        self.lower = lower
        self.world = world
        self.timeout = timeout  # number of steps before the middle layer should give up

    def do(self,plan):
        """carry out actions.
        actions is of the form {'visit':list_of_locations}
        It visits the locations in turn.
        """
        to_do = plan['visit']
        for loc in to_do:
            position = self.world.locations[loc]
            arrived = self.lower.do({'go_to':position, 'timeout':self.timeout})
            self.display(1,"Goal",loc,arrived)

from agentEnv import Rob_body, World

def rob_ex():
    global world, body, middle, top
    world = World(walls = {((20,0),(30,20)), ((70,-5),(70,25))},
                  locations = {'mail':(-5,10), 
                              'o103':(50,10), 'o109':(100,10),'storage':(101,51)})
    body = Rob_body(world)
    middle = Rob_middle_layer(body)
    top = Rob_top_layer(middle, world)

# try:
# top.do({'visit':['o109','storage','o109','o103']})
# You can directly control the middle layer:
# middle.do({'go_to':(30,-5), 'timeout':200})
# Can you make it go around in circles?
# Can you make it crash?

if __name__ == "__main__":
    rob_ex()
    print("Try: top.do({'visit':['o109','storage','o109','o103']})")

# Robot Trap for which the current controller cannot escape:
def robot_trap():
    global trap_world, trap_body, trap_middle, trap_top
    trap_world = World({((10, 51), (60, 51)), ((30, 10), (30, 20)),
                            ((10, -1), (10, 20)), ((10, 30), (10, 51)),
                            ((30, 30), (30, 40)), ((10, -1), (60, -1)),
                            ((10, 30), (30, 30)), ((10, 20), (30, 20)),
                            ((60, -1), (60, 51))},
                     locations={'goal':(90,25)})
    trap_body = Rob_body(trap_world,init_pos=(0,25), init_dir=90)
    trap_middle = Rob_middle_layer(trap_body)
    trap_top = Rob_top_layer(trap_middle, trap_world)

# Robot trap exercise:
# robot_trap()
# trap_body.do({'steer':'straight'})
# trap_top.do({'visit':['goal']})
# What if the goal was further to the right?

