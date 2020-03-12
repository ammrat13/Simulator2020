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
          fov=62.2,
          aspect=16/9,
          nearVal=0.1,
          farVal=3.1)
        self.camera_h_res = 640
        self.camera_v_res = 360
        self.camera_focal_len = .1

    def load_urdf(self):
        """Load the URDF of the blockstacker into the environment

        The blockstacker URDF comes with its own dimensions and
        textures, collidables.
        """
        # Note that the position is set later in this method
        self.robot = p.loadURDF(Utilities.gen_urdf_path("blockstacker/urdf/blockstacker.urdf"),
                                [0, 0, 0], [0, 0, 0, 1], useFixedBase=False)

        p.setJointMotorControlMultiDof(self.robot,
                                       self.caster_link,
                                       p.POSITION_CONTROL,
                                       [0, 0, 0],
                                       targetVelocity=[100000, 100000, 100000],
                                       positionGain=0,
                                       velocityGain=1,
                                       force=[0, 0, 0])

        p.setJointMotorControlArray(self.robot, self.flywheel_links, p.VELOCITY_CONTROL,
                                    targetVelocities=[-2, 2],
                                    forces=[1, 1])

        # Set it again because loadURDF uses URDF coordinates not COM
        self.set_pose((-.73, 0, 0))

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
        
    def __posort_to_pose__(self, pos, ort):
        # Just use a utility method to extrapolate the SIP
        p_pos = p.multiplyTransforms(pos, ort, [0,COM_TO_SIP,0], [0,0,0,1])[0][0:2]
        a_pos = p.multiplyTransforms(pos, ort, [0,COM_TO_AXE,0], [0,0,0,1])[0][0:2]
        p_theta = atan2(p_pos[1]-a_pos[1], p_pos[0]-a_pos[0])
        return (*p_pos, p_theta)

    def get_pose(self, NOISE_POS=.02, NOISE_ANG=10):
        # Get the pose
        r_pos, r_ort = p.getBasePositionAndOrientation(self.robot)
        pose = list(self.__posort_to_pose__(r_pos, r_ort))
        # We have to edit since we add noise
        pose[0] += gauss(0, NOISE_POS)
        pose[1] += gauss(0, NOISE_POS)
        pose[2] += vonmisesvariate(0, NOISE_ANG)
        # Return as needed
        return tuple(pose)

    def set_pose(self, pose, SPAWN_Z=.05):
        # Use the utility method
        p.resetBasePositionAndOrientation(self.robot,
            *self.__pose_to_posort__(pose, SPAWN_Z=SPAWN_Z))
        return self.get_pose()

    def read_wheel_velocities(self, noisy=True):
        # TODO - implement noisy
        noise = 0.0
        rmotor, lmotor = p.getJointStates(self.robot, self.motor_links)
        # print("positions ", rmotor[0], lmotor[0])
        return (rmotor[1] + noise, lmotor[1] + noise)

    def command_wheel_velocities(self, rtarget_vel, ltarget_vel):
        self.drive.rtarget_vel = rtarget_vel
        self.drive.ltarget_vel = ltarget_vel
        return self.read_wheel_velocities()

    def capture_image(self, camera_num=0):
        *_, camera_position, camera_orientation = p.getLinkState(self.robot, self.camera_links[camera_num])
        camera_look_position, _ = p.multiplyTransforms(camera_position, camera_orientation, [0,0.1,0], [0,0,0,1])
        view_matrix = p.computeViewMatrix(
          cameraEyePosition=camera_position,
          cameraTargetPosition=camera_look_position,
          cameraUpVector=(0, 0, 1))
        return p.getCameraImage(
          self.camera_h_res,
          self.camera_v_res,
          view_matrix,
          self.camera_projection_matrix,
          renderer=p.ER_BULLET_HARDWARE_OPENGL
        )[2]

    def capture_images(self, poses, MAKE_TRANSPARENT_BEFORE=True, MAKE_VISIBLE_AFTER=True):
        # The threshold at which objects become invisible for this method
        # Is a weird non-linear function:
        #   trueDepth = far * near / (far - (far-near) * this_value)
        # Algebra gives `1` for what the mask should be, but that doesn't 
        #   match observation
        # Set it to a value close to one
        DEPTH_MASK = .99

        # If we need to, make outselves transparent
        # Store the old colors for later
        old_colors = {}
        if MAKE_TRANSPARENT_BEFORE:
            temp = p.getVisualShapeData(self.robot)
            for ji in range(-1, p.getNumJoints(self.robot)):
                old_colors[ji] = temp[ji+1][7]
                p.changeVisualShape(self.robot, ji, rgbaColor=[0,0,0,0])

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

        # Make ourselves visible again if we need to
        if MAKE_VISIBLE_AFTER:
            for ji, color in old_colors.items():
                p.changeVisualShape(self.robot, ji, rgbaColor=color)

        return ret

    def step(self):
        self.drive.step(self.robot, self.enabled)

        if not self.enabled:
            p.changeVisualShape(self.robot, self.button_link, rgbaColor=[1, 1, self.blink, 1])
            self.blink_count += 1
            if self.blink_count > 40:
                self.blink = not self.blink
                self.blink_count = 0
        else:
            p.changeVisualShape(self.robot, self.button_link, rgbaColor=[1, 1, 0, 1])
