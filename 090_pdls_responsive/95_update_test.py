########## INIT ####################################################################################

##### Imports #####

### Standard ###
import sys, time, math
now = time.time
from traceback import print_exc, format_exc
from pprint import pprint

### Special ###
import numpy as np
from py_trees.common import Status


### Local ###

## PDDLStream ##
sys.path.append( "../pddlstream/" )
from pddlstream.algorithms.meta import solve
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import read, INF, get_file_path
from pddlstream.language.constants import print_solution, PDDLProblem

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



########## HELPER FUNCTIONS ########################################################################





########## EXECUTIVE (THE METHOD) ##################################################################

class UpdateTest:
    """ Least structure needed to compare plans """

    ##### Init ############################################################

    def reset_beliefs( self ):
        """ Erase belief memory """
        self.memory  = ObjectMemory() # Distributions over objects
        self.last    = {}


    def reset_state( self ):
        """ Erase problem state """
        self.symbols  = list()
        self.status   = Status.INVALID
        self.facts    = list()
        self.goal     = tuple()
        self.task     = None
        self.currPlan = None
        self.action   = None


    def __init__( self, world = None ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.world = world if (world is not None) else PB_BlocksWorld()
        self.reset_beliefs()
        self.reset_state()
        self.logger = DataLogger()
        # DEATH MONITOR
        self.noSoln =  0
        self.nonLim = 10


    def set_table( self ):
        """ Get ready for an experiment """
        self.world.robot.goto_home()
        self.world.reset_blocks()
        self.world.spin_for( 500 )


    def phase_1_Perceive( self, Nscans ):
        """ Take in evidence and form beliefs """

        for _ in range( Nscans ):
            self.memory.belief_update( self.world.full_scan_noisy() ) # We need at least an initial set of beliefs in order to plan
            print( f"There are {len(self.memory.beliefs)} beliefs" )
            for bel in self.memory.beliefs:
                print( f"\t{bel.labels}" )
            print( '\n' )


        objs = self.memory.scan_consistent()

        print( f"Starting Objects:" )
        for obj in objs:
            print( f"\t{obj}" )

        self.symbols = objs
        self.status  = Status.RUNNING


########## MAIN ####################################################################################
if __name__ == "__main__":
    planner = UpdateTest()
    planner.set_table()
    planner.phase_1_Perceive( 20 )