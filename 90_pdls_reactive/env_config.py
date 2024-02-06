########## INIT ####################################################################################
import os, math
import numpy as np
import pybullet_data



########## PATHS ###################################################################################

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
TABLE_URDF_PATH = os.path.join( pybullet_data.getDataPath(), "table/table.urdf" )



########## OBJECTS #################################################################################

_NULL_NAME    = "NOTHING"
# _BLOCK_NAMES  = ['redBlock', 'ylwBlock', 'bluBlock', 'grnBlock', 'ornBlock', 'vioBlock', _NULL_NAME]
_BLOCK_NAMES  = ['redBlock', _NULL_NAME]
_POSE_DIM     = 7
_ACTUAL_NAMES = _BLOCK_NAMES[:-1]
_SUPPORT_NAME = 'table'
_BLOCK_SCALE  = 0.038
_MIN_X_OFFSET = 0.400



########## UR5 ROBOT ###############################################################################

_GRASP_VERT_OFFSET = _BLOCK_SCALE * 2.0
_GRASP_ORNT_XYZW   = np.array( [0, 0.7070727, 0, 0.7071408,] )
_Q_HOME            = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]
_ROT_VEL_SMALL     = 0.005



########## BELIEFS #################################################################################

_N_POSE_UPDT = 25
_POSN_STDDEV =  0.008
_NULL_THRESH =  0.75



########## PLANNER #################################################################################

_LOG_PROB_FACTOR = 10.0
_LOG_BASE        =  2.0
_PLAN_THRESH     =  0.02
_ACCEPT_POSN_ERR =  0.5*_BLOCK_SCALE 
_PROB_TICK_FAIL = 0.01 # 0.20
_Z_SAFE         = 8.0*_BLOCK_SCALE