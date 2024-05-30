import math, os
from collections import namedtuple
from datetime import datetime
from random import random

from attrdict import AttrDict
import pybullet as pb
import numpy as np

from roboticstoolbox.robot.ERobot import ERobot

from utils import homog_to_pb_posn_ornt, p_lst_has_nan

from env_config import ( ROBOT_URDF_PATH, _ROT_VEL_SMALL, _Q_HOME, _MIN_X_OFFSET, _BLOCK_SCALE, 
                         _BASE_POSN, _BASE_ORNT, )

class UR5e_RBT( ERobot ):
    """ PyRBT for UR5e IK """

    def __init__( self ):
        """ Load URDF from local files """
        
        links, name, urdf, path = self.URDF_read( 
            # os.path.join( os.getcwd(), ROBOT_URDF_PATH[2:] )
            ROBOT_URDF_PATH
        )
        super().__init__(
            links,
            name         = name.upper(),
            manufacturer = 'Universal Robots',
        )
        
        self.urdfDesc = urdf
        self.srcPath  = path

    def fk_posn_ornt( self, joint_angles ):
        """ Get forward kinematics in [Px,Py,Pz],[Ox,Oy,Oz,Ow] """
        return homog_to_pb_posn_ornt( np.array( self.fkine( joint_angles, end = "tool0" ) ) )


class UR5Sim:
    """ Original Author: Josep Daniel, https://github.com/josepdaniel/ur5-bullet/tree/master/UR5 """

    def load_robot( self ):
        """ Load UR5e from description """
        flags = 0 #pb.URDF_USE_SELF_COLLISION
        robot = pb.loadURDF( ROBOT_URDF_PATH, _BASE_POSN, _BASE_ORNT, flags=flags )
        return robot
  
    def __init__( self ):
        """ Load robot and controller """
        
        self.end_effector_index = 7 # 5 # 6 # 7
        self.jntIndices = [1,2,3,4,5,6]
        self.ur5 = self.load_robot()
        self.kinMdl = UR5e_RBT()
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

        self.Nfails =  0
        self.Nlimit =  3
        self.lastT  = self.get_current_pose()
        self.name   = "UR5e"

    def fk_posn_ornt( self, joint_angles ):
        """ Get forward kinematics in [Px,Py,Pz],[Ox,Oy,Oz,Ow] """
        return self.kinMdl.fk_posn_ornt( joint_angles )

    def get_name( self ):
        """ Return name """
        return self.name

    def set_joint_angles( self, joint_angles ):
        """ Set a joint-space goal and pursue it with maximum force """
        poses   = []
        indexes = []
        forces  = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append( joint.maxForce ) #*0.85 )

        pb.setJointMotorControlArray(
            self.ur5, indexes,
            pb.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            # positionGains=[0.04]*len(poses), forces=forces
            positionGains=[0.06]*len(poses), forces=forces
        )

    def get_joint_vel( self ):
        """ Return the current joint velocities """
        j = pb.getJointStates( self.ur5, self.jntIndices )
        return [i[1] for i in j]
    
    # def get_joint_acc( self ):
    #     """ Return the current joint velocities """
    #     j = pb.getJointStates( self.ur5, self.jntIndices )
    #     return [i[2] for i in j]
    
    def p_moving( self ):
        """ Return True if any of the joint velocities are above some small number """
        vel = self.get_joint_vel()
        if p_lst_has_nan( vel ):
            vel = [0.0 for _ in range(6)]
        # while p_lst_has_nan( vel ):
        #     pb.stepSimulation()
        #     vel = self.get_joint_vel()
        # print( f"Joint Vel: {vel}" )
        for v in vel:
            if abs( v ) > _ROT_VEL_SMALL:
                return True
        return False

    def get_joint_angles( self ):
        """ Return the current joint configuration """
        j = pb.getJointStates( self.ur5, self.jntIndices )
        return [i[0] for i in j]

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
        lower_limits = [-2*math.pi]*6
        upper_limits = [2*math.pi]*6
        joint_ranges = [4*math.pi]*6
        rest_poses   = _Q_HOME
        joint_angles = pb.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
    

    def calculate_ik_quat( self, position, quaternion ):
        """ Get the target joint angles to achieve the desired `position` and `orientation` (Quaternion) """
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        # rest_poses   = _Q_HOME
        rest_poses   = self.get_joint_angles()
        joint_angles = pb.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.1]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses,
            maxNumIterations  = 2000,
            residualThreshold = 0.005
        )
        return joint_angles
    
    def goto_home( self ):
        """ Go to the home config """
        self.set_joint_angles( _Q_HOME )

    def goto_pb_posn_ornt( self, posn, ornt ):
        """ Set the target joint angles to achieve the desired `position` and `orientation` (Quaternion) """
        q = self.calculate_ik_quat( posn, ornt )
        self.set_joint_angles( q )
        
    def get_current_pose( self ):
        """ Get the current pose in the lab frame """
        linkstate = pb.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = list(linkstate[0]), list(linkstate[1])
        return position, orientation
    
    def anger( self, posn, ornt ):
        scl = 2.0
        posn = np.add( posn, [ 
            _MIN_X_OFFSET + random()*scl*_BLOCK_SCALE, 
            random()*2.0*scl*_BLOCK_SCALE-scl*_BLOCK_SCALE, 
            0.5*_BLOCK_SCALE 
        ] )
        self.goto_pb_posn_ornt( posn, ornt )

    def log_fail( self, posn, ornt ):
        self.Nfails += 1
        if self.Nfails >= self.Nlimit:
            self.Nfails = 0
            self.anger( posn, ornt )

if __name__ == "__main__":
    foo = UR5e_RBT()
    print( foo.fk_posn_ornt( [0.0 for _ in range(6)] ) )