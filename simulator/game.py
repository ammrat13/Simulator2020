#!/usr/bin/env python3
"""
File:          Game.py
Author:        Alex Cui
Last Modified: Binit on 10/30
"""

import os
import time
import pybullet as p

from simulator.field import Field
from simulator.legos import Legos
from simulator.utilities import Utilities
from simulator.trainingbot_agent import TrainingBotAgent

class Game:
    """Maintains and coordinates the game loop"""

    def __init__(self):
        """Initializes game elements
        Sets up game and simulation elements. For example,
        the three minute timer of the game.
        """
        self.cwd = os.getcwd()
        self.cid = p.connect(p.SHARED_MEMORY)
        if (self.cid < 0):
            p.connect(p.GUI)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # For video recording (works best on Mac and Linux, not well on Windows)
        #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
        
        # Don't do realtime simulation
        # We will manually step
        p.setRealTimeSimulation(0)

        self.agent = TrainingBotAgent()
        self.field = Field()
        self.legos = Legos()

    def load_statics(self):
        """Loading the static objects
        Including field, buttons, and more.
        """
        self.field.load_urdf(self.cwd)
        self.legos.load_lego_urdfs(self.cwd, [(0, .3, "#ffffff")])        

    def load_agents(self):
        """Loading the agents
        Including the button robot and the mobile block stacking robot.
        """
        self.agent.load_urdf(self.cwd)

    def load_ui(self):
        """Loading the UI components
        Such as sliders or buttons.
        """
        self.maxForceSlider = p.addUserDebugParameter("maxForce", 0, 5, 1)

    def read_ui(self):
        """Reads the UI components' state
        And publishes them for all of game to process
        """
        maxForce = p.readUserDebugParameter(self.maxForceSlider)
        self.agent.set_max_force(maxForce)

    def process_keyboard_events(self):
        """Read keyboard events
        And publishes them for all of game to process
        """
        keys = p.getKeyboardEvents()

        if keys.get(65296): #right
            self.agent.keyboard_control = True
            self.agent.increaseRTargetVel()
            self.agent.decreaseLTargetVel()
        elif keys.get(65295): #left
            self.agent.keyboard_control = True
            self.agent.increaseLTargetVel()
            self.agent.decreaseRTargetVel()
        elif keys.get(65297): #up
            self.agent.keyboard_control = True
            self.agent.increaseLTargetVel()
            self.agent.increaseRTargetVel()
        elif keys.get(65298): #down
            self.agent.keyboard_control = True
            self.agent.decreaseLTargetVel()
            self.agent.decreaseRTargetVel()
        # Only interfere if we are controlling with keys
        # Otherwise, the agent will handle stopping
        elif self.agent.keyboard_control:
            self.agent.normalizeLTargetVel()
            self.agent.normalizeRTargetVel()

    def monitor_buttons(self):
        # Store the return values for readability
        buttonStates = p.getJointStates(
                            self.field.model_id,
                            [b.joint_id for b in self.field.buttons])

        # Get every button and press it if needed
        for i,x in enumerate(buttonStates):
            if x[0] < -.0038:
                self.field.buttons.press_button(i)
            else:
                self.field.buttons.unpress_button(i)

        # We don't have logic changing the button color
        # Too costly in terms of time
        # Can easily be implemented because logic there

    def run(self):
        """Maintains the game loop
        Coordinates other functions to execute here and
        tracks the delta time between each game loop.
        """
        self.load_statics()
        self.load_agents()
        self.load_ui()

        while True:
            old_time = time.time()

            self.read_ui()
            self.process_keyboard_events()

            self.monitor_buttons()
            self.field.buttons.update(1/240)

            self.agent.update()

            # Steps time by 1/240 seconds
            p.stepSimulation()

            # Sleep to 1/240 seconds if we need to
            if 1/240 - (time.time() - old_time) > 0:
                time.sleep(1/240 - (time.time() - old_time))
