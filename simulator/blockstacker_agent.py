#!/usr/bin/env python3
"""
File:          blockstacker_agent.py
Author:        Binit Shah 
Last Modified: Binit on 3/2
"""

import pybullet as p
from math import atan2

from simulator.differentialdrive import DifferentialDrive
from simulator.utilities import Utilities
from planning2020 import planning

started = False
state = 0

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

    def load_urdf(self):
        """Load the URDF of the blockstacker into the environment

        The blockstacker URDF comes with its own dimensions and
        textures, collidables.
        """
        self.robot = p.loadURDF(Utilities.gen_urdf_path("blockstacker/urdf/blockstacker.urdf"),
                                [-.85, 0.025, 0.05], [0, 0, -.707, .707], useFixedBase=False)

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

    def get_pose(self):
        # TODO - fix orientation
        r_pos, r_ort = p.getBasePositionAndOrientation(self.robot)
        p_pos = p.multiplyTransforms(r_pos, r_ort, [0,.174676,0], [0,0,0,1])[0]
        p_theta = atan2(p_pos[1]-r_pos[1], p_pos[0]-r_pos[0])
        return (p_pos[0], p_pos[1], p_theta)

    def set_pose(self, pose):
        # TODO - fix orientation
        p.resetBasePositionAndOrientation(self.robot, [pose[0], pose[1], 0.07], [0, 0, -.707, .707])
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

    def plan(self):
        global started
        global state

        if not started:
            started = True
            if state == 0:
                planning.queue_bin(self.get_pose(), 1, .3)
            if state == 1:
                planning.queue_bin(self.get_pose(), 6, .3)
            if state == 2:
                planning.queue_bin(self.get_pose(), 8, .3)
            if state == 3:
                planning.queue_bin(self.get_pose(), 2, .3)
            if state == 4:
                planning.queue_end(self.get_pose())
        elif planning.wp_done():
            started = False
            state += 1
        else:
            self.command_wheel_velocities(*planning.compute_wheel_velocities(self.get_pose()))

    def capture_image(self):
        # Camera
        *_, camera_position, camera_orientation = p.getLinkState(self.robot, self.camera_link)
        camera_look_position, _ = p.multiplyTransforms(camera_position, camera_orientation, [0,0.1,0], [0,0,0,1])
        view_matrix = p.computeViewMatrix(
          cameraEyePosition=camera_position,
          cameraTargetPosition=camera_look_position,
          cameraUpVector=(0, 0, 1))
        projection_matrix = p.computeProjectionMatrixFOV(
          fov=45.0,
          aspect=1.0,
          nearVal=0.1,
          farVal=3.1)
        return p.getCameraImage(300, 300, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]

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
