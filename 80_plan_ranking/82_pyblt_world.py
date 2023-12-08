import time
from random import random, choice
from pprint import pprint

import numpy as np

import pybullet as p
import pybullet_data

from spatialmath import SO3
from spatialmath.base import tr2angvec, r2q
from spatialmath.quaternion import UnitQuaternion

from scipy.stats import chi2


########## UTILITY FUNCTIONS #######################################################################

def pb_posn_ornt_to_homog( posn, ornt ):
    """ Express the PyBullet position and orientation as homogeneous coords """
    H = np.eye(4)
    Q = UnitQuaternion( ornt[-1], ornt[:3] )
    H[0:3,0:3] = Q.SO3().R
    H[0:3,3]   = np.array( posn )
    return H


def pb_posn_ornt_to_row_vec( posn, ornt ):
    """ Express the PyBullet position and orientation as a Position and Orientation --> [Px,Py,Pz,Ow,Ox,Oy,Oz] """
    V = list( posn[:] )
    V.append( ornt[-1] )
    V.extend( ornt[:3] )
    return np.array(V)


def homog_to_row_vec( homog ):
    """ Express a homogeneous coord as a Position and Orientation --> [Px,Py,Pz,Ow,Ox,Oy,Oz] """
    P = homog[0:3,3]
    Q = UnitQuaternion( r2q( homog[0:3,0:3] ) )
    V = np.zeros( (7,) )
    V[0:3] = P[:]
    V[3]   = Q.s
    V[4:7] = Q.v[:]
    return np.array(V)


def total_pop( odds ):
    """ Sum over all categories in the prior odds """
    total = 0
    for k in odds:
        total += odds[k]
    return total


def normalize_dist( odds_ ):
    """ Normalize the distribution so that the sum equals 1.0 """
    total  = total_pop( odds_ )
    rtnDst = dict()
    for k in odds_:
        rtnDst[k] = odds_[k] / total
    return rtnDst


def roll_outcome( odds ):
    """ Get a random outcome from the distribution """
    oddsNorm = normalize_dist( odds )
    distrib  = []
    outcome  = []
    total    = 0.0
    for o, p in oddsNorm.items():
        total += p
        distrib.append( total )
        outcome.append( o )
    roll = random()
    for i, p in enumerate( distrib ):
        if roll <= p:
            return outcome[i]
    return None



########## UTILITY CLASSES & SYMBOLS ###############################################################
_POSN_STDDEV = 0.008


class SimpleBlock:
    """ Use this to ground the RYB blocks """
    def __init__( self, name, pcd, pose ):
        self.name = name
        self.pcd  = pcd
        self.pose = pose


class ObjectSymbol:
    """ Determinized object """
    def __init__( self, ref, label, pose ):
        """ Assign members """
        self.ref   = ref
        self.label = label
        self.pose  = pose
    def __repr__( self ):
        """ String representation """
        return f"<{self.label} @ {self.pose}, P={self.ref.labels[self.label]}>"


class ObjectBelief:
    """ Hybrid belief: A discrete distribution of classes that may exist at a continuous distribution of poses """

    def __init__( self, initStddev = _POSN_STDDEV ):
        """ Initialize with origin poses and uniform, independent variance """
        stdDev = [initStddev if (i<3) else 0.0 for i in range(7)]
        self.labels  = {} # ---------------------- Current belief in each class
        self.pose    = np.array([0,0,0,1,0,0,0]) # Mean pose
        self.pStdDev = np.array(stdDev) # -------- Pose variance
        self.pHist   = [] # ---------------------- Recent history of poses
        self.pThresh = 0.5 # --------------------- Minimum prob density at which a nearby pose is relevant
        self.covar   = np.zeros( (7,7,) ) # ------ Pose covariance matrix
        for i, stdDev in enumerate( self.pStdDev ):
            self.covar[i,i] = stdDev * stdDev

    def get_posn( self, poseOrBelief ):
        """ Get the position from the object """
        if isinstance( poseOrBelief, ObjectBelief ):
            return poseOrBelief.pose[0:3]
        elif isinstance( poseOrBelief, np.ndarray ):
            if poseOrBelief.size == (4,4,):
                return poseOrBelief[0:3,3]
            else:
                return poseOrBelief[0:3]

    def p_pose_relevant( self, poseOrBelief ):
        """ Determine if a nearby pose is relevant """
        x        = self.get_posn( poseOrBelief )
        mu       = self.get_posn( self.pose )
        sigma    = self.covar[0:3,0:3]
        m_dist_x = np.dot((x-mu).transpose(),np.linalg.inv(sigma))
        m_dist_x = np.dot(m_dist_x, (x-mu))
        return (1-chi2.cdf( m_dist_x, 3 ) >= self.pThresh)
    
    def sample_symbol( self ):
        """ Sample a determinized symbol from the hybrid distribution """
        label = roll_outcome( self.labels )
        return ObjectSymbol( 
            self,
            label, 
            np.random.multivariate_normal( self.pose, self.covar ) 
        )
    
    



########## ENVIRONMENT #############################################################################
_BLOCK_NAMES  = ['redBlock', 'ylwBlock', 'bluBlock', 'grnBlock', 'ornBlock', 'vioBlock',]
_SUPPORT_NAME = 'table'

class PB_BlocksWorld:
    """ Simple physics simulation with 3 blocks """

    def __init__( self ):
        """ Create objects """
        self.physicsClient = p.connect( p.GUI ) # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath( pybullet_data.getDataPath() ) #optionally
        p.setGravity( 0, 0, -10 )
        self.planeId = p.loadURDF( "plane.urdf" )

        redBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        p.changeVisualShape( redBlock, -1, rgbaColor=[1.0, 0.0, 0.0, 1] )

        ylwBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        p.changeVisualShape( ylwBlock, -1, rgbaColor=[1.0, 1.0, 0.0, 1] )

        bluBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        p.changeVisualShape( bluBlock, -1, rgbaColor=[0.0, 0.0, 1.0, 1] )

        grnBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        p.changeVisualShape( grnBlock, -1, rgbaColor=[0.0, 1.0, 0.0, 1] )

        ornBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        p.changeVisualShape( ornBlock, -1, rgbaColor=[1.0, 0.5, 0.0, 1] )

        vioBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        p.changeVisualShape( vioBlock, -1, rgbaColor=[0.5, 0.0, 1.0, 1] )

        self.blocks = [redBlock, ylwBlock, bluBlock, grnBlock, ornBlock, vioBlock,]

        for i in range( 100 ):
            p.stepSimulation()

    def step( self ):
        """ Advance one step and sleep """
        p.stepSimulation()
        time.sleep( 1.0 / 240.0 )

    def stop( self ):
        """ Disconnect from the simulation """
        p.disconnect()

    def get_blocks_RYB( self ):
        """ Find the RYB blocks, Fully Observable """
        rtnBlocks = [None,None,None]
        for i, name in enumerate( _BLOCK_NAMES ):
            blockPos, blockOrn = p.getBasePositionAndOrientation( self.blocks[i] )
            pose = pb_posn_ornt_to_homog( blockPos, blockOrn )
            rtnBlocks[i] = SimpleBlock( name, None, pose )
        return rtnBlocks
    
    def get_block_noisy( self, blockName, confuseProb = 0.10, poseStddev = _POSN_STDDEV ):
        """ Find one of the RYB blocks, Partially Observable, Return None if the name is not in the world """
        try:
            idx = _BLOCK_NAMES.index( blockName )
            blockPos, blockOrn = p.getBasePositionAndOrientation( self.blocks[idx] )
            blockPos = np.array( blockPos ) + np.array( [np.random.normal( 0.0, poseStddev/3.0 ) for _ in range(3)] )
            rtnObj = ObjectBelief()
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


########## MOCK PLANNER ############################################################################

class MockAction:
    """ Least Behavior """
    def __init__( self, objName, dest ):
        self.objName = objName # - Type of object required
        self.dest    = dest # ---- Where we will place this object
        self.status  = "INVALID" # Current status of this behavior
        self.symbol  = None # ---- Symbol on which this behavior relies

class MockPlanner:
    """ Least structure needed to compare plans """
    def __init__( self ):
        self.poses = { # Intended destinations
            "P1" : [ 0.300,0.000,0.150,1,0,0,0],
            "P2" : [ 0.600,0.000,0.150,1,0,0,0],
            "P3" : [ 0.450,0.000,0.300,1,0,0,0],
            "P4" : [-0.300,0.000,0.150,1,0,0,0],
            "P5" : [-0.600,0.000,0.150,1,0,0,0],
            "P6" : [-0.450,0.000,0.300,1,0,0,0],
        }
        self.skltns = [ # Plan skeletons, Each builds an arch
            [MockAction('redBlock',self.poses['P1']),MockAction('ylwBlock',self.poses['P2']),MockAction('bluBlock',self.poses['P3']),],
            [MockAction('grnBlock',self.poses['P4']),MockAction('ornBlock',self.poses['P5']),MockAction('vioBlock',self.poses['P6']),],
        ]

########## MAIN ####################################################################################
##### Env. Settings #####
np.set_printoptions( precision = 3, linewidth = 145 )

if __name__ == "__main__":

    # print( homog_to_row_vec( np.eye(4) ) )

    # ods = {
    #     'a': 25,
    #     'b': 50,
    #     'c': 25
    # }
    # res = {}
    # for k in ods.keys():
    #     res[k] = 0

    # for _ in range(100000):
    #     res[ roll_outcome( ods ) ] += 1
    # pprint( res )

    # bName = "redBlock"
    
    # block = world.get_block_noisy( bName )

    world = PB_BlocksWorld()
    objs  = []
    for name in _BLOCK_NAMES:
        objs.append( world.get_block_noisy( name ) )

    for i in range(3):
        for obj in objs:
            print( obj.sample_symbol() )
        print()
        
    #         print( .pose[:3] )
    #     print()
    # world.get_blocks_RYB()

