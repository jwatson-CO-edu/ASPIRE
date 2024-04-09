########## INIT ####################################################################################
from random import random

from utils import row_vec_to_pb_posn_ornt
from env_config import ( TABLE_URDF_PATH, _MIN_X_OFFSET, _BLOCK_SCALE, _CONFUSE_PROB, )



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

########## SIMULATED VISION ########################################################################

class ObjectReading:
    """ Represents a signal coming from the vision pipeline """
    def __init__( self, distribution = None, objPose = None ):
        self.dstr = distribution if (distribution is not None) else {} # Current belief in each class
        self.pose = objPose if isinstance(objPose, (list,np.ndarray)) else np.array([0,0,0,1,0,0,0]) # Object pose


class NoisyObjectSensor:
    """ A fake vision pipeline with simple discrete class error and Gaussian pose error """

    def __init__( self, confuseProb = _CONFUSE_PROB ):
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