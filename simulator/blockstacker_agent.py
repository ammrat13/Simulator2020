#!/usr/bin/env python3
"""
File:          blockstacker_agent.py
Author:        Binit Shah 
Last Modified: Binit on 3/2
"""

import pybullet as p
from math import atan2
from random import gauss, vonmisesvariate

from simulator.differentialdrive import DifferentialDrive
from simulator.utilities import Utilities
from planning2020 import planning

COM_TO_SIP = .174676
COM_TO_AXE = .074676

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

    def get_pose(self, NOISE_POS=0.02, NOISE_ANG=10):
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

    def plan(self, t):
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
                planning.queue_bin(self.get_pose(), 5, .3)
            if state == 5:
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
