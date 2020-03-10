#!/usr/bin/env python3
"""
File:          blockstacker_agent.py
Author:        Binit Shah 
Last Modified: Binit on 3/2
"""

import pybullet as p

from simulator.differentialdrive import DifferentialDrive
from simulator.utilities import Utilities

class BlockStackerAgent:
    """The BlockStackerAgent class maintains the blockstacker agent"""
    def __init__(self, vel_delta=0.5, skew=0.0):
        """Setups infomation about the agent
        """
        self.camera_links = [6, 8]
        self.motor_links = [10, 12]
        self.flywheel_links = [14, 16]
        self.stepper_link = 1
        self.button_link = 4
        self.caster_link = 18
        self.tower_link = 2

        self.drive = DifferentialDrive(self.motor_links, max_force=0.2, vel_limit=6.0, vel_delta=vel_delta, skew=skew)

        self.enabled = True
        self.blink = 0
        self.blink_count = 0
