########## INIT ####################################################################################

import os
import math 
import numpy as np
import time
import pybullet as pb
import random
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
TABLE_URDF_PATH = os.path.join( pybullet_data.getDataPath(), "table/table.urdf" )



########## DEV PLAN ################################################################################
"""
[Y] Load table, 2024-01-25: Compied from example, No GUI
[Y] Load robot, 2024-01-25: Compied from example, No GUI
[ ] World class
[ ] Load blocks
[ ] Pick block
"""



########## HELPER FUNCTIONS ########################################################################

def start_GUI():
    """ Connect to the GUI and start real time sim """
    pb.connect(pb.GUI)
    pb.setRealTimeSimulation( True )

def make_table():
    """ Load a table """
    # table = pb.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
    pb.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])



########## UR5 ROBOT ###############################################################################

class UR5Sim:
    """ Original Author: Josep Daniel, https://github.com/josepdaniel/ur5-bullet/tree/master/UR5 """

    def load_robot( self ):
        """ Load UR5 from description """
        flags = pb.URDF_USE_SELF_COLLISION
        robot = pb.loadURDF( ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags )
        return robot
  
    def __init__( self, camera_attached=False ):
        """ Load robot and controller """
        
        self.end_effector_index = 7
        self.ur5 = self.load_robot()
        self.num_joints = pb.getNumJoints( self.ur5 )
        
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range( self.num_joints ):
            info = pb.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pb.setJointMotorControl2(self.ur5, info.id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info     

    def set_joint_angles( self, joint_angles ):
        """ Set a joint-space goal and pursue it with maximum force """
        poses   = []
        indexes = []
        forces  = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pb.setJointMotorControlArray(
            self.ur5, indexes,
            pb.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )


    def get_joint_angles( self ):
        """ Return the current joint configuration """
        j = pb.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    

    def check_collisions( self, verbose = False ):
        """ Return True if there is a self-collision """
        collisions = pb.getContactPoints()
        if len(collisions) > 0:
            if verbose:
                print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False


    def calculate_ik_euler( self, position, orientation ):
        """ Get the target joint angles to achieve the desired `position` and `orientation` (Euler Angles) """
        quaternion   = pb.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses   = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]
        joint_angles = pb.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
       

    def add_gui_sliders( self ):
        """ Display GUI pose control """
        self.sliders = []
        self.sliders.append(pb.addUserDebugParameter("X", 0, 1, 0.4))
        self.sliders.append(pb.addUserDebugParameter("Y", -1, 1, 0))
        self.sliders.append(pb.addUserDebugParameter("Z", 0.3, 1, 0.4))
        self.sliders.append(pb.addUserDebugParameter("Rx", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pb.addUserDebugParameter("Ry", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pb.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, 0))


    def read_gui_sliders( self ):
        """ Get user pose input """
        x = pb.readUserDebugParameter(self.sliders[0])
        y = pb.readUserDebugParameter(self.sliders[1])
        z = pb.readUserDebugParameter(self.sliders[2])
        Rx = pb.readUserDebugParameter(self.sliders[3])
        Ry = pb.readUserDebugParameter(self.sliders[4])
        Rz = pb.readUserDebugParameter(self.sliders[5])
        return [x, y, z, Rx, Ry, Rz]
        
    def get_current_pose( self ):
        """ Get the current pose in the lab frame """
        linkstate = pb.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return position, orientation
    


########## MAIN ####################################################################################

if __name__ == "__main__":
    start_GUI()
    make_table()
    robot = UR5Sim()
    for i in range( 2000 ):
        pb.stepSimulation()
        time.sleep( 1.0 / 240.0 )