########## INIT ####################################################################################
import os, math
import numpy as np
import pybullet_data



########## PATHS ###################################################################################

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
TABLE_URDF_PATH = os.path.join( pybullet_data.getDataPath(), "table/table.urdf" )



########## OBJECTS #################################################################################

_NULL_NAME    = "NOTHING"
_BLOCK_NAMES  = ['redBlock', 'ylwBlock', 'bluBlock', 'grnBlock', 'ornBlock', 'vioBlock', _NULL_NAME]
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