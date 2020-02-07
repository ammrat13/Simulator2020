#!/usr/bin/env python3
"""
File:          trainingbot_agent.py
Author:        Binit Shah
Last Modified: Binit on 12/11
"""

from math import cos, sin, sqrt

import pybullet as p

from simulator.utilities import Utilities

class TrainingBotAgent:
    """The TrainingBotAgent class maintains the trainingbot agent"""
    def __init__(self, motion_delta=0.1):
        """Setups infomation about the agent
        """
        self.camera_link = 15
        self.caster_links = [12, 13]
        self.motor_links = [3, 8]

        # Differential motor control
        self.max_force = 1
        self.motion_delta = motion_delta
        self.velocity_limit = 5
        # As fractions of self.velocity_limit
        # Range from -1.0 to 1.0
        self.ltarget_vel, self.rtarget_vel = 0, 0
        self.vel_gen = self.generate_target_velocities()

        # Start autonomous and switch when needed
        self.keyboard_control = False

    def load_urdf(self, cwd):
        """Load the URDF of the trainingbot into the environment

        The trainingbot URDF comes with its own dimensions and
        textures, collidables.
        """
        self.robot = p.loadURDF(Utilities.gen_urdf_path("trainingbot/urdf/trainingbot.urdf", cwd), [-0.93, 0, 0.1], [0.5, 0.5, 0.5, 0.5], useFixedBase=False)
        p.setJointMotorControlArray(self.robot, self.caster_links, p.VELOCITY_CONTROL, targetVelocities=[0, 0], forces=[0, 0])

    def increaseLTargetVel(self):
        self.ltarget_vel += self.motion_delta
        if self.ltarget_vel >= 1.0:
            self.ltarget_vel = 1.0

    def decreaseLTargetVel(self):
        self.ltarget_vel -= self.motion_delta
        if self.ltarget_vel <= -1.0:
            self.ltarget_vel = -1.0

    def normalizeLTargetVel(self):
        if self.ltarget_vel < -self.motion_delta:
            self.ltarget_vel += self.motion_delta
        elif self.ltarget_vel > self.motion_delta:
            self.ltarget_vel -= self.motion_delta

    def increaseRTargetVel(self):
        self.rtarget_vel += self.motion_delta
        if self.rtarget_vel >= 1.0:
            self.rtarget_vel = 1.0

    def decreaseRTargetVel(self):
        self.rtarget_vel -= self.motion_delta
        if self.rtarget_vel <= -1.0:
            self.rtarget_vel = -1.0

    def normalizeRTargetVel(self):
        if self.rtarget_vel < -self.motion_delta:
            self.rtarget_vel += self.motion_delta
        elif self.rtarget_vel > self.motion_delta:
            self.rtarget_vel -= self.motion_delta

    def set_max_force(self, max_force):
        self.max_force = max_force

    def update(self):
        # Guidance
        # Only do this if we are not being manually controlled
        if not self.keyboard_control:
            self.rtarget_vel, self.ltarget_vel = next(self.vel_gen)

        # Movement
        p.setJointMotorControlArray(
            self.robot,
            self.motor_links,
            p.VELOCITY_CONTROL,
            targetVelocities=[
                self.velocity_limit * self.rtarget_vel,
                self.velocity_limit * self.ltarget_vel],
            forces=[self.max_force, self.max_force])

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
        #p.getCameraImage(300, 300, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    def generate_target_velocities(self):
        R = .035
        L = .1
        D = .22

        BEZIERX = [-.93,2,-1,0]
        BEZIERY = [0,0,1,-2]

        currentTheta = 0.0
        currentU = 0.0

        lastCtrlPt = p.multiplyTransforms(
                        p.getBasePositionAndOrientation(self.robot)[0],
                        p.getBasePositionAndOrientation(self.robot)[1],
                        (0,0,L),
                        (0,0,0,1))[0]

        lid = p.addUserDebugLine(
                        [BEZIERX[0]+BEZIERX[1]*currentU+BEZIERX[2]*currentU**2+BEZIERX[3]*currentU**3,
                         BEZIERY[0]+BEZIERY[1]*currentU+BEZIERY[2]*currentU**2+BEZIERY[3]*currentU**3,
                         -10],
                        [BEZIERX[0]+BEZIERX[1]*currentU+BEZIERX[2]*currentU**2+BEZIERX[3]*currentU**3,
                         BEZIERY[0]+BEZIERY[1]*currentU+BEZIERY[2]*currentU**2+BEZIERY[3]*currentU**3,
                         10])

        while True:
            jstates = p.getJointStates(self.robot, self.motor_links)
            wl = jstates[0][1]
            wr = jstates[1][1]

            xDotC0 = R/2 * cos(currentTheta) - R*L/D * sin(currentTheta)
            xDotC1 = R/2 * cos(currentTheta) + R*L/D * sin(currentTheta)
            yDotC0 = R/2 * sin(currentTheta) + R*L/D * cos(currentTheta)
            yDotC1 = R/2 * sin(currentTheta) - R*L/D * cos(currentTheta)

            xDot = xDotC0*wr + xDotC1*wl
            yDot = yDotC0*wr + yDotC1*wl
            thetaDot = (R/D)*wr + (-R/D)*wl

            currentTheta += thetaDot / 240

            newCtrlPt = p.multiplyTransforms(
                            p.getBasePositionAndOrientation(self.robot)[0],
                            p.getBasePositionAndOrientation(self.robot)[1],
                            (0,0,L),
                            (0,0,0,1))[0]
            myxDot = 240 * (newCtrlPt[0] - lastCtrlPt[0])
            myyDot = 240 * (newCtrlPt[1] - lastCtrlPt[1])
            lastCtrlPt = newCtrlPt

            dl = sqrt(xDot**2 + yDot**2) / 240
            xDotTarg = 3*BEZIERX[3] * currentU**2 + 2*BEZIERX[2] * currentU + BEZIERX[1]
            yDotTarg = 3*BEZIERY[3] * currentU**2 + 2*BEZIERY[2] * currentU + BEZIERY[1]
            currentU += dl / sqrt(xDotTarg**2 + yDotTarg**2)


            matDetInv = 1 / (xDotC0*yDotC1 - xDotC1*yDotC0)
            wrTarg = yDotC1*matDetInv*xDotTarg - xDotC1*matDetInv*yDotTarg
            wlTarg = -yDotC0*matDetInv*xDotTarg + xDotC0*matDetInv*yDotTarg

            wmax = max(abs(wrTarg), abs(wlTarg))
            wrTarg /= max(1, wmax)
            wlTarg /= max(1, wmax)

            p.removeUserDebugItem(lid);
            lid = p.addUserDebugLine(
                        [BEZIERX[0]+BEZIERX[1]*currentU+BEZIERX[2]*currentU**2+BEZIERX[3]*currentU**3,
                         BEZIERY[0]+BEZIERY[1]*currentU+BEZIERY[2]*currentU**2+BEZIERY[3]*currentU**3,
                         -10],
                        [BEZIERX[0]+BEZIERX[1]*currentU+BEZIERX[2]*currentU**2+BEZIERX[3]*currentU**3,
                         BEZIERY[0]+BEZIERY[1]*currentU+BEZIERY[2]*currentU**2+BEZIERY[3]*currentU**3,
                         10])

            print(f"{xDot - myxDot} {yDot - myyDot}")

            yield (wlTarg, wrTarg)
