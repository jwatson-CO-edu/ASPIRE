########## INIT ####################################################################################

##### Imports #####

### Standard ###
import sys, time, math
now = time.time
from time import sleep
from traceback import print_exc, format_exc
from pprint import pprint
from random import random

### Special ###
import numpy as np
from py_trees.common import Status


### Local ###

## MAGPIE ##
sys.path.append( "../" )
# from magpie.poses import translation_diff

from utils import ( row_vec_to_pb_posn_ornt, pb_posn_ornt_to_row_vec, diff_norm, closest_dist_Q_to_segment_AB,
                    DataLogger, )

from env_config import ( _GRASP_VERT_OFFSET, _GRASP_ORNT_XYZW, _NULL_NAME, _ACTUAL_NAMES, _MIN_X_OFFSET,
                         _NULL_THRESH, _BLOCK_SCALE, _CLOSEST_TO_BASE, _ACCEPT_POSN_ERR, _MIN_SEP, _Z_SAFE,
                         _N_POSE_UPDT, _WP_NAME, _SAMPLE_DET )

from PB_BlocksWorld import PB_BlocksWorld, rand_table_pose
from symbols import Object, Path

from beliefs import ObjectMemory
from actions import Plan, display_PDLS_plan, BT_Runner, get_ith_BT_action_from_PDLS_plan, Place, Stack

from spatialmath import Quaternion
from spatialmath.quaternion import UnitQuaternion


########## HELPER FUNCTIONS ########################################################################

def pb_posn_ornt_in_frame( posn, ornt, fPosn, fOrnt ):
    """ Get `posn`, `ornt` WRT `fPosn`, `fOrnt` """
    rPosn = np.subtract( posn, fPosn )
    Qf    = UnitQuaternion( fOrnt[-1], fOrnt[:3] )
    Qq    = UnitQuaternion( ornt[-1] , ornt[:3]  )
    Qi    = Qf.inv()
    Qr    = Qi * Qq
    rOrnt = list()
    rOrnt.append( Qr.s )
    rOrnt.extend( Qr.v.tolist() )
    return rPosn, rOrnt


def calc_grasp_from_row_vec( objPose ):
    """ A function that returns grasps """
    grasp_pose = list( objPose )
    grasp_pose[2] += _GRASP_VERT_OFFSET
    posn, _ = row_vec_to_pb_posn_ornt( grasp_pose )
    ornt = _GRASP_ORNT_XYZW.copy()
    # return pb_posn_ornt_to_row_vec( posn, ornt )
    return posn, ornt


def perturb_pb_orientation( ornt, delta_rad ):
    """ Perturb the `ornt` up to `delta_rad` """
    Qq     = UnitQuaternion( ornt[0], ornt[1:]  )
    dTheta = -delta_rad + random()*2.0*delta_rad
    if random() < 0.5:
        Qr = UnitQuaternion.Rx( dTheta )
    else:
        Qr = UnitQuaternion.Ry( dTheta )
    Qt = Qq * Qr  
    rOrnt = list()
    rOrnt.append( Qt.s )
    rOrnt.extend( Qt.v.tolist() )
    return rOrnt



########## TESTING #################################################################################

class UR5_Test_Harness:
    """ Test UR5 Config """

    ##### Init ############################################################

    def __init__( self, world = None ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.world = world if (world is not None) else PB_BlocksWorld()

    def setup( self ):
        """ Get ready for an experiment """
        self.world.robot.goto_home()
        self.world.spin_for( 500 )



########## MAIN ####################################################################################
if __name__ == "__main__":
    testEnv = UR5_Test_Harness()
    testEnv.setup()
    sleep( 5 )

    while 1:
        testEnv.world.reset_blocks()
        blocks = testEnv.world.full_scan_true()
        for block in blocks:
            print( block.pose )
            bPosn, bOrnt = calc_grasp_from_row_vec( block.pose )
            bOrnt = perturb_pb_orientation( bOrnt, np.pi/4.0 )
            testEnv.world.robot.goto_pb_posn_ornt( bPosn, bOrnt )
            testEnv.world.spin_for( 200 )
            sleep( 0.25 )
        