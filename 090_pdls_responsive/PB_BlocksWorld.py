import time
from random import random

import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client as bc
import pybullet_data

from UR5Sim import UR5Sim
from utils import row_vec_to_pb_posn_ornt, pb_posn_ornt_to_row_vec
from symbols import Object
from beliefs import ObjectBelief
from env_config import ( _MIN_X_OFFSET, _BLOCK_SCALE, TABLE_URDF_PATH, _BLOCK_NAMES, _POSN_STDDEV, 
                         _USE_GRAPHICS, _ACCEPT_POSN_ERR, _ACTUAL_NAMES, _ONLY_RED, _ONLY_PRIMARY,
                         _BLOCK_ALPHA, _CONFUSE_PROB )

##### Paths #####


########## HELPERS #################################################################################

class DummyBelief:
    """ Stand-in for an actual `ObjectBelief` """
    def __init__( self, label ):
        self.labels = { label: 1.0 }

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


def banished_pose():
    """ Send it to the shadow realm """
    return [100,100,100], [0, 0, 0, 1]

def make_block( clientRef ):
    """ Load a block at the correct scale, place it random, and return the int handle """
    posn, _ = rand_table_pose()
    return clientRef.loadURDF( 
        "cube.urdf", 
        posn,
        globalScaling = 0.25/4.0
    )

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

class ObjectReading:
    """ Represents a signal coming from the vision pipeline """
    def __init__( self, distribution = None, objPose = None ):
        self.dstr = distribution if (distribution is not None) else {} # Current belief in each class
        self.pose = objPose if isinstance(objPose, (list,np.ndarray)) else np.array([0,0,0,1,0,0,0]) # Object pose


class NoisyObjectSensor:
    """ A fake vision pipeline with simple discrete class error and Gaussian pose error """

    def __init__( self, poseStddev = _POSN_STDDEV, confuseProb = _CONFUSE_PROB ):
        self.pStdDev  = np.array( poseStddev ) 
        self.confProb = confuseProb

    def noisy_reading_from_true( self, trueObj ):
        """ Add noise to the true reading and return it """
        rtnObj = ObjectReading()
        blockPos, blockOrn = row_vec_to_pb_posn_ornt( trueObj.pose )
        blockPos = np.array( blockPos ) + np.array( [np.random.normal( 0.0, self.pStdDev[i] ) for i in range(3)] )
        rtnObj.pose = pb_posn_ornt_to_row_vec( blockPos, blockOrn )
        for blkName_i in _BLOCK_NAMES:
            if blkName_i == trueObj.label:
                rtnObj.dstr[ blkName_i ] = 1.0-self.confProb*(len( _BLOCK_NAMES )-1)
            else:
                rtnObj.dstr[ blkName_i ] = self.confProb
        return rtnObj

########## ENVIRONMENT #############################################################################


class PB_BlocksWorld:
    """ Simple physics simulation with blocks """

    ##### Init ############################################################

    def __init__( self, graphicsOverride = False ):
        """ Create objects """
        ## Init Sim ##
        self.period        = 1.0 / 240.0
        if _USE_GRAPHICS or graphicsOverride:
            # self.physicsClient = pb.connect( pb.GUI ) # or p.DIRECT for non-graphical version
            self.physicsClient = bc.BulletClient( connection_mode = pb.GUI ) # or p.DIRECT for non-graphical version
        else:
            # self.physicsClient = pb.connect( pb.DIRECT )
            self.physicsClient = bc.BulletClient( connection_mode = pb.DIRECT )
        self.physicsClient.setAdditionalSearchPath( pybullet_data.getDataPath() ) #optionally
        self.physicsClient.setGravity( 0, 0, -10 )

        ## Instantiate Robot and Table ##
        self.table     = make_table( self.physicsClient )
        self.robot     = UR5Sim()
        self.robotName = "UR5e"
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
        ePsn, _    = self.robot.get_current_pose()
        pDif = np.subtract( bPsn, ePsn )
        self.grasp.append( (hndl,pDif,bOrn,) ) # Preserve the original orientation because I am lazy


    def robot_grasp_at( self, graspPose ):
        """ Lock the block to the end effector that is nearest the effector """
        hndl = self.get_handle_at_pose( graspPose, 2.0*_ACCEPT_POSN_ERR )
        if hndl is not None:
            blockName = self.get_handle_name( hndl )
            symb = self.get_block_true( blockName )
            bPsn, bOrn = row_vec_to_pb_posn_ornt( symb.pose )
            ePsn, _    = self.robot.get_current_pose()
            pDif = np.subtract( bPsn, ePsn )
            self.grasp.append( (hndl,pDif,bOrn,) ) # Preserve the original orientation because I am lazy


    def robot_release_all( self ):
        """ Unlock all objects from end effector """
        self.grasp = list()


    ##### Block Queries ###################################################

    def get_handle( self, name ):
        """ Get the ID of the requested object by `name` """
        if name in _BLOCK_NAMES:
            return self.blocks[ _BLOCK_NAMES.index( name ) ]
        else:
            return None
        
    def get_handle_at_pose( self, rowVec, posnErr = _POSN_STDDEV*2.0 ):
        """ Return the handle of the object nearest to the `rowVec` pose if it is within `posnErr`, Otherwise return `None` """
        posnQ, _ = row_vec_to_pb_posn_ornt( rowVec )
        distMin = 1e6
        indxMin = -1
        for i, blk in enumerate( self.blocks ):
            if blk is not None:
                blockPos, _ = self.physicsClient.getBasePositionAndOrientation( blk )
                dist = np.linalg.norm( np.array( posnQ ) - np.array( blockPos ) )
                if dist < distMin:
                    distMin = dist
                    indxMin = i
        if (indxMin > -1) and (distMin <= posnErr):
            return self.blocks[ indxMin ]
        return None
    

    def get_handle_name( self, handle ):
        """ Get the block name that corresponds to the handle """
        try:
            idx = self.blocks.index( handle )
            return _BLOCK_NAMES[ idx ]
        except ValueError:
            return None


    def get_block_true( self, blockName ):
        """ Find one of the ROYGBV blocks, Fully Observable, Return None if the name is not in the world """
        try:
            idx = _BLOCK_NAMES.index( blockName )
            blockPos, blockOrn = self.physicsClient.getBasePositionAndOrientation( self.blocks[idx] )
            # blockPos = np.array( blockPos )
            return Object( 
                blockName, 
                pb_posn_ornt_to_row_vec( blockPos, blockOrn ),
                DummyBelief( blockName ), 
            )
        except ValueError:
            return None
        
        
    def full_scan_true( self ):
        """ Find all of the ROYGBV blocks, Fully Observable """
        rtnSym = []
        for name in _ACTUAL_NAMES:
            rtnSym.append( self.get_block_true( name ) )
        return rtnSym
    

    ##### Simulation ######################################################

    def step( self ):
        """ Advance one step and sleep """
        self.physicsClient.stepSimulation()
        time.sleep( self.period )
        ePsn, _    = self.robot.get_current_pose()
        for obj in self.grasp:
            self.physicsClient.resetBasePositionAndOrientation( obj[0], np.add( obj[1], ePsn ), obj[2] )

    def spin_for( self, N = 1000 ):
        """ Run for `N` steps """
        for _ in range(N):
            self.step()

    def stop( self ):
        """ Disconnect from the simulation """
        self.physicsClient.disconnect()
    

    ##### Pose Sampling ###################################################

    def get_block_noisy( self, blockName, confuseProb = 0.10, poseStddev = _POSN_STDDEV ):
        """ Find one of the ROYGBV blocks, Partially Observable, Return None if the name is not in the world """
        try:
            rtnObj = ObjectBelief()
            idx = _BLOCK_NAMES.index( blockName )
            blockPos, blockOrn = self.physicsClient.getBasePositionAndOrientation( self.blocks[idx] )
            blockPos = np.array( blockPos ) + np.array( [np.random.normal( 0.0, poseStddev/3.0 ) for _ in range(3)] )
            rtnObj.pose = pb_posn_ornt_to_row_vec( blockPos, blockOrn )
            for i in range( len( _BLOCK_NAMES ) ):
                blkName_i = _BLOCK_NAMES[i]
                if blkName_i == blockName:
                    rtnObj.labels[ blkName_i ] = 1.0-confuseProb*(len( _BLOCK_NAMES )-1)
                else:
                    rtnObj.labels[ blkName_i ] = confuseProb
            return rtnObj
        except ValueError:
            return None
        

    def full_scan_noisy( self, confuseProb = 0.10, poseStddev = _POSN_STDDEV ):
        """ Find all of the ROYGBV blocks, Partially Observable """
        rtnBel = []
        for name in _ACTUAL_NAMES:
            rtnBel.append( self.get_block_noisy( name, confuseProb, poseStddev ) )
        return rtnBel
    

    def check_predicate( self, symbol, posnErr = _POSN_STDDEV*2.0 ):
        """ Check that the `symbol` is True """
        handle = self.get_handle_at_pose( symbol.pose, posnErr )
        return (self.get_handle_name( handle ) == symbol.label)
    

    def validate_goal_spec( self, spec, posnErr = _POSN_STDDEV*2.0 ):
        """ Return true only if all the predicates in `spec` are true """
        for p in spec:
            if not self.check_predicate( p, posnErr ):
                return False
        return True