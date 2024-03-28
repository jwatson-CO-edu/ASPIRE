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