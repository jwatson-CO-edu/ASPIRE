########## INIT ####################################################################################
from random import random

import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client as bc
import pybullet_data

from symbols import GraspObj
from utils import row_vec_to_pb_posn_ornt
from env_config import ( TABLE_URDF_PATH, _MIN_X_OFFSET, _BLOCK_SCALE, _CONFUSE_PROB, _BLOCK_NAMES,
                         _USE_GRAPHICS, _BLOCK_ALPHA, _ONLY_PRIMARY, _ONLY_RED )



########## HELPERS #################################################################################

def make_table( clientRef ):
    """ Load a table """
    # table = pb.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
    return clientRef.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])


def rand_table_pose():
    """ Return a random pose in the direct viscinity if the robot """
    return [ 
        _MIN_X_OFFSET + random()*10.0*_BLOCK_SCALE, 
        random()*20.0*_BLOCK_SCALE-10.0*_BLOCK_SCALE, 
        _BLOCK_SCALE 
    ], [0, 0, 0, 1]


def make_block( clientRef ):
    """ Load a block at the correct scale, place it random, and return the int handle """
    posn, _ = rand_table_pose()
    return clientRef.loadURDF( 
        "cube.urdf", 
        posn,
        globalScaling = 0.25/4.0
    )


def banished_pose():
    """ Send it to the shadow realm """
    return [100,100,100], [0, 0, 0, 1]


def draw_cross( clientRef, position, scale, color, w = 2.0 ):
    """ Draw a static cross at the XYZ `position` with arms aligned with the lab axes """
    ofst = scale/2.0
    cntr = np.array( position )
    Xdif = [ofst,0,0]
    Ydif = [0,ofst,0]
    Zdif = [0,0,ofst]
    Xlin = [ np.add( cntr, Xdif ), np.subtract( cntr, Xdif ) ]
    Ylin = [ np.add( cntr, Ydif ), np.subtract( cntr, Ydif ) ]
    Zlin = [ np.add( cntr, Zdif ), np.subtract( cntr, Zdif ) ]
    clientRef.addUserDebugLine( Xlin[0], Xlin[1], color, lineWidth = w )
    clientRef.addUserDebugLine( Ylin[0], Ylin[1], color, lineWidth = w )
    clientRef.addUserDebugLine( Zlin[0], Zlin[1], color, lineWidth = w )


########## SIMULATED VISION ########################################################################

class ObjectReading( GraspObj ):
    """ Represents a signal coming from the vision pipeline """
    def __init__( self, labels = None, pose = None ):
        """ Init distribution and  """
        super().__init__( None, pose )
        self.labels = labels if (labels is not None) else {} # Current belief in each class
        


class NoisyObjectSensor:
    """ A fake vision pipeline with simple discrete class error and Gaussian pose error """

    def __init__( self, confuseProb = _CONFUSE_PROB ):
        self.confProb = confuseProb

    def noisy_reading_from_true( self, trueObj ):
        """ Add noise to the true reading and return it """
        rtnObj = ObjectReading()
        rtnObj.pose = np.array( trueObj.pose )
        for blkName_i in _BLOCK_NAMES:
            if blkName_i == trueObj.label:
                rtnObj.labels[ blkName_i ] = 1.0-self.confProb*(len( _BLOCK_NAMES )-1)
            else:
                rtnObj.labels[ blkName_i ] = self.confProb
        return rtnObj
    

########## ROBOT ###################################################################################

class GhostRobot:
    """ Floating effector without physics or extent """

    def __init__( self, initPose = None ):
        """ Set the initial location of the effector """
        self.pose   = np.array( initPose ) if isinstance(initPose, (list,np.ndarray)) else np.array([0,0,0,1,0,0,0])
        self.target = np.array( self.pose )
        self.speed  = _BLOCK_SCALE / 10.0

    def tick( self ):
        """ Advance the cursor by one speed """
        bgn = self.pose[:3]
        end = self.target[:3]
        dif = np.subtract( end, bgn )
        mag = np.linalg.norm( dif )
        if mag > 0.0:
            unt = dif / mag
        else:
            unt = np.zeros( (3,) )
        if mag > self.speed:
            dif = unt * self.speed
            self.pose[:3] = np.add( bgn, dif )
        else:
            self.pose = np.array( self.target )

    def draw( self, clientRef ):
        """ Render the effector cursor """
        draw_cross( clientRef, self.pose[:3], _BLOCK_SCALE*4.0, [0,0,0], w = 2.0 )



########## ENVIRONMENT #############################################################################


class PB_BlocksWorld:
    """ Simple physics simulation with blocks """

    ##### Init ############################################################

    def __init__( self, graphicsOverride = False ):
        """ Create objects """

        ## Init Sim ##
        self.period        = 1.0 / 240.0
        if _USE_GRAPHICS or graphicsOverride:
            self.physicsClient = bc.BulletClient( connection_mode = pb.GUI ) # or p.DIRECT for non-graphical version
        else:
            self.physicsClient = bc.BulletClient( connection_mode = pb.DIRECT )
        self.physicsClient.setAdditionalSearchPath( pybullet_data.getDataPath() ) #optionally
        self.physicsClient.setGravity( 0, 0, -10 )

        ## Instantiate Robot and Table ##
        self.table     = make_table( self.physicsClient )
        self.robot     = GhostRobot()
        self.robotName = "Ghost"
        self.grasp     = []

        ## Instantiate Blocks ##
        redBlock = make_block( self.physicsClient )
        self.physicsClient.changeVisualShape( redBlock, -1, rgbaColor=[1.0, 0.0, 0.0, _BLOCK_ALPHA] )

        ylwBlock = make_block( self.physicsClient )
        self.physicsClient.changeVisualShape( ylwBlock, -1, rgbaColor=[1.0, 1.0, 0.0, _BLOCK_ALPHA] )

        bluBlock = make_block( self.physicsClient )
        self.physicsClient.changeVisualShape( bluBlock, -1, rgbaColor=[0.0, 0.0, 1.0, _BLOCK_ALPHA] )

        grnBlock = make_block( self.physicsClient )
        self.physicsClient.changeVisualShape( grnBlock, -1, rgbaColor=[0.0, 1.0, 0.0, _BLOCK_ALPHA] )

        ornBlock = make_block( self.physicsClient )
        self.physicsClient.changeVisualShape( ornBlock, -1, rgbaColor=[1.0, 0.5, 0.0, _BLOCK_ALPHA] )

        vioBlock = make_block( self.physicsClient )
        self.physicsClient.changeVisualShape( vioBlock, -1, rgbaColor=[0.5, 0.0, 1.0, _BLOCK_ALPHA] )

        self.blocks = [redBlock, ylwBlock, bluBlock, grnBlock, ornBlock, vioBlock, None]

        ## Fake Vision ##
        self.sensor = NoisyObjectSensor()

    ##### Block Movements #################################################

    def reset_blocks( self ):
        """ Send blocks to random locations """
        for blockHandl in self.blocks:
            if blockHandl is not None:
                posn, ornt = rand_table_pose()
                self.physicsClient.resetBasePositionAndOrientation( blockHandl, posn, ornt )
                if _ONLY_PRIMARY and blockHandl not in [self.get_handle( nam ) for nam in ['redBlock','ylwBlock','bluBlock']]:
                    posn, ornt = banished_pose()
                    self.physicsClient.resetBasePositionAndOrientation( blockHandl, posn, ornt )
                elif _ONLY_RED and (blockHandl != self.get_handle( 'redBlock' )):
                    posn, ornt = banished_pose()
                    self.physicsClient.resetBasePositionAndOrientation( blockHandl, posn, ornt )

    def robot_grasp_block( self, blockName ):
        """ Lock the block to the end effector """
        hndl = self.get_handle( blockName )
        symb = self.get_block_true( blockName )
        bPsn, bOrn = row_vec_to_pb_posn_ornt( symb.pose )
        ePsn = self.robot.pose[:3]
        pDif = np.subtract( bPsn, ePsn )
        self.grasp.append( (hndl,pDif,bOrn,) ) # Preserve the original orientation because I am lazy

    # FIXME, START HERE: GRASP THE NEAREST BLOCK