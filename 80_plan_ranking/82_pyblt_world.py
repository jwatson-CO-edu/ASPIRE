import time
from random import random, choice

import numpy as np

import pybullet as p
import pybullet_data

# from spatialmath import SO3
# from spatialmath.base import tr2angvec, r2q
from spatialmath.quaternion import UnitQuaternion


########## UTILITY FUNCTIONS #######################################################################

def pb_posn_ornt_to_homog( posn, ornt ):
    """ Express the PyBullet position and orientation as homogeneous coords """
    H = np.eye(4)
    Q = UnitQuaternion( ornt[-1], ornt[:3] )
    H[0:3,0:3] = Q.SO3().R
    H[0:3,3]   = np.array( posn )
    return H
    


########## UTILITY CLASSES #########################################################################

class SimpleBlock:
    """ Use this to ground the RYB blocks """
    def __init__( self, name, pcd, pose ):
        self.name = name
        self.pcd  = pcd
        self.pose = pose

########## ENVIRONMENT #############################################################################
_BLOCK_NAMES  = ['redBlock', 'ylwBlock', 'bluBlock',]
_SUPPORT_NAME = 'table'

class PB_BlocksWorld:
    """ Simple physics simulation with 3 blocks """

    def __init__( self ):
        """ Create objects """
        self.physicsClient = p.connect( p.GUI ) # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath( pybullet_data.getDataPath() ) #optionally
        p.setGravity( 0, 0, -10 )
        self.planeId  = p.loadURDF( "plane.urdf" )

        redBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        p.changeVisualShape( redBlock, -1, rgbaColor=[1.0, 0.0, 0.0, 1] )

        ylwBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        p.changeVisualShape( ylwBlock, -1, rgbaColor=[1.0, 1.0, 0.0, 1] )

        bluBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        p.changeVisualShape( bluBlock, -1, rgbaColor=[0.0, 0.0, 1.0, 1] )

        self.blocks = [redBlock, ylwBlock, bluBlock,]

        for i in range( 100 ):
            p.stepSimulation()

    def run( self ):
        """ Run for a bit """
        for i in range( 2000 ):
            p.stepSimulation()
            time.sleep( 1.0 / 240.0 )
        p.disconnect()

    def get_blocks_RYB( self ):
        """ Find the RYB blocks, Fully Observable """
        rtnBlocks = [None,None,None]
        for i, name in enumerate( _BLOCK_NAMES ):
            blockPos, blockOrn = p.getBasePositionAndOrientation( self.blocks[i] )
            pose = pb_posn_ornt_to_homog( blockPos, blockOrn )
            rtnBlocks[i] = SimpleBlock( name, None, pose )
        return rtnBlocks
    
    def get_blocks_RYB_noisy( self, confuseProb = 0.20, poseStddev = 0.125 ):
        """ Find the RYB blocks, Partially Observable """
        rtnBlocks = [None,None,None]
        for i, name in enumerate( _BLOCK_NAMES ):
            blockPos, blockOrn = p.getBasePositionAndOrientation( self.blocks[i] )
            pose = pb_posn_ornt_to_homog( blockPos, blockOrn )
            rtnBlocks[i] = SimpleBlock( name, None, pose )
            if random() < confuseProb:
                rtnBlocks[i].name = choice( _BLOCK_NAMES )
            offset = np.zeros( (4,4) )
        return rtnBlocks


########## MAIN ####################################################################################
if __name__ == "__main__":
    world = PB_BlocksWorld()
    world.get_blocks_RYB()