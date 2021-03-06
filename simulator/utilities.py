#!/usr/bin/env python3
"""
File:          Utilities.py
Author:        Mridul Bansal
Last Modified: Binit on 10/12
"""

import os
import numpy as np
import pybullet as p

class Utilities:
    """Utility class to convert between units of measure"""

    def gen_urdf_path(urdf_fname):
        _SIM_ROOT = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(_SIM_ROOT, "..", "data", urdf_fname)

    def draw_debug_pose(position=np.array([0,0,0]),
                        orientation=np.array([0,0,0,1]),
                        x_color=np.array([1,0,0]),
                        y_color=np.array([0,1,0]),
                        z_color=np.array([0,0,1]),
                        lineLength=0.1,
                        lineWidth=1,
                        lifeTime=0,
                        parentObjectUniqueId=-1,
                        parentLinkIndex=-1,
                        replaceItemUniqueIds=(-1, -1, -1)):
        '''Create a pose marker that identifies a position and orientation in space with 3 colored lines.
        '''
        pts = np.array([[0,0,0],[lineLength,0,0],[0,lineLength,0],[0,0,lineLength]])
        rotIdentity = np.array([0,0,0,1])
        po, _ = p.multiplyTransforms(position, orientation, pts[0,:], rotIdentity)
        px, _ = p.multiplyTransforms(position, orientation, pts[1,:], rotIdentity)
        py, _ = p.multiplyTransforms(position, orientation, pts[2,:], rotIdentity)
        pz, _ = p.multiplyTransforms(position, orientation, pts[3,:], rotIdentity)
        px_uid = p.addUserDebugLine(po, px, x_color, lineWidth, lifeTime, parentObjectUniqueId, parentLinkIndex, replaceItemUniqueIds[0])
        py_uid = p.addUserDebugLine(po, py, y_color, lineWidth, lifeTime, parentObjectUniqueId, parentLinkIndex, replaceItemUniqueIds[1])
        pz_uid = p.addUserDebugLine(po, pz, z_color, lineWidth, lifeTime, parentObjectUniqueId, parentLinkIndex, replaceItemUniqueIds[2])
        return (px_uid, py_uid, pz_uid)

    def draw_debug_info(time=None, points=None, etc=None,
                        position=np.array([-0.05,1.0,0]),
                        color=np.array([1,1,1]),
                        textSize=2,
                        lifeTime=0,
                        parentObjectUniqueId=-1,
                        parentLinkIndex=-1,
                        replaceItemUniqueId=-1):
        '''Draw text for time, points, and other debug information
        '''

        time_msg = '{0:02.0f}:{1:02.0f}'.format(*divmod(time, 60)) if time else ""
        points_msg = f" | points: {points}" if points else ""
        etc_msg = f"| etc" if etc else ""
        msg = f"{time_msg}{points_msg}{etc_msg}"

        m_uid = p.addUserDebugText(msg, textPosition=position,
                                        textColorRGB=color,
                                        textSize=textSize,
                                        lifeTime=lifeTime,
                                        parentObjectUniqueId=parentObjectUniqueId,
                                        parentLinkIndex=parentLinkIndex,
                                        replaceItemUniqueId=replaceItemUniqueId)

        return m_uid

    def draw_bezier_curve(BEZIERX,
                        BEZIERY,
                        steps = 100,
                        z = 0.1,
                        lineWidth=1,
                        lifeTime=0,
                        parentObjectUniqueId=-1,
                        parentLinkIndex=-1,
                        replaceItemUniqueIds = -1):
        curve = []
        for i in range(steps):
            u_initial = i/steps
            u_final = (i + 1)/steps
            x_initial = BEZIERX[0] + BEZIERX[1]*u_initial + BEZIERX[2]*(u_initial**2) + BEZIERX[3]*(u_initial**3)
            x_final = BEZIERX[0] + BEZIERX[1]*u_final + BEZIERX[2]*(u_final**2) + BEZIERX[3]*(u_final**3)
            y_initial = BEZIERY[0] + BEZIERY[1]*u_initial + BEZIERY[2]*(u_initial**2) + BEZIERY[3]*(u_initial**3)
            y_final = BEZIERY[0] + BEZIERY[1]*u_final + BEZIERY[2]*(u_final**2) + BEZIERY[3]*(u_final**3)
            p_initial = [x_initial, y_initial, z]
            p_final = [x_final, y_final, z]
            curve.append(p.addUserDebugLine(p_initial, p_final, np.array([1,0,0]), lineWidth, lifeTime, parentObjectUniqueId, parentLinkIndex, replaceItemUniqueIds))
        return curve

    def print_multibody_links(body_id):
        for link_id in range(p.getNumJoints(body_id)):
            print(p.getJointInfo(body_id, link_id))
