#!/usr/bin/env python3
"""
File:          blockstacker_agent.py
Author:        Binit Shah 
Last Modified: Binit on 3/2
"""

import pybullet as p
import numpy as np
from math import sin, cos, atan2
from random import gauss, vonmisesvariate
from operator import sub

from simulator.differentialdrive import DifferentialDrive
from simulator.utilities import Utilities

COM_TO_SIP = .174676
COM_TO_AXE = .074676
# We center the camera on the x-axis
# Old offset was [.0807,.0324,.06624]
CAM_OFFSET_VEC = [0,.0324,.06624]

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

        self.camera_projection_matrix = p.computeProjectionMatrixFOV(
          fov=45.0,
          aspect=1.0,
          nearVal=0.1,
          farVal=3.1)
        self.camera_h_res = 300
        self.camera_v_res = 300
        self.camera_focal_len = .1

    # Utility methods for converting between representations
    def __pose_to_posort__(self, pose, SPAWN_Z=.05):
        # Position is easy -- we are given X, Y, Z. Just remember it is the 
        #   position of the single integrator point, not the robot itself
        # For theta, we can use axis angle, but remember default orientation 
        #   is .707 - .707k, not identity
        # We can use axis angle to get the desired quaternion
        # Orientation is (cos(t/2) + sin(t/2)k) * (.707 - .707k)
        return (
            (pose[0] - COM_TO_SIP*cos(pose[2]), pose[1] - COM_TO_SIP*sin(pose[2]), SPAWN_Z),
            (0, 0, .707 * (sin(pose[2]/2) - cos(pose[2]/2)), .707 * (sin(pose[2]/2) + cos(pose[2]/2)))
        )

    def capture_images(self, poses, MAKE_TRANSPARENT_BEFORE=True, MAKE_VISIBLE_AFTER=True):
        # The threshold at which objects become invisible for this method
        # Is a weird non-linear function:
        #   trueDepth = far * near / (far - (far-near) * this_value)
        # Algebra gives `1` for what the mask should be, but that doesn't 
        #   match observation
        # Set it to a value close to one
        DEPTH_MASK = .999

        ret = []
        for po in poses:
            # Compute camera position and orientation
            c_pos, c_ort = self.__pose_to_posort__(po)
            c_pos, _ = p.multiplyTransforms(c_pos, c_ort, CAM_OFFSET_VEC, [0,0,0,1])
            c_lop, _ = p.multiplyTransforms(c_pos, c_ort, [0, self.camera_focal_len, 0], [0,0,0,1])
            # Actually capture the image using similar logic to `capture_image`
            img, depth = p.getCameraImage(
              self.camera_h_res,
              self.camera_v_res,
              p.computeViewMatrix(c_pos, c_lop, (0,0,1)),
              self.camera_projection_matrix,
              flags=p.ER_NO_SEGMENTATION_MASK
            )[2:4]
            # Transparancy with depth
            np.place(img[:,:,3], depth>DEPTH_MASK, 0)
            # Return
            ret.append(img)

        return ret
