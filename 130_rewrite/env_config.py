########## INIT ####################################################################################
import os
import numpy as np
import pybullet_data



########## SETTINGS ################################################################################

ROBOT_URDF_PATH = os.path.expanduser( "~/MAGPIE/090_pdls_responsive/ur_e_description/urdf/ur5e.urdf" )
TABLE_URDF_PATH = os.path.join( pybullet_data.getDataPath(), "table/table.urdf" )
_VERBOSE        = True



########## PYBULLET ################################################################################

_USE_GRAPHICS = True
_BLOCK_ALPHA  = 1.0



########## OBJECTS #################################################################################

_NULL_NAME    = "NOTHING"
_ONLY_RED     = False
_ONLY_PRIMARY = False
if _ONLY_RED:
    _BLOCK_NAMES  = ['redBlock', _NULL_NAME]
elif _ONLY_PRIMARY:
    _BLOCK_NAMES  = ['redBlock', 'ylwBlock', 'bluBlock', _NULL_NAME]
else:
    _BLOCK_NAMES  = ['redBlock', 'ylwBlock', 'bluBlock', 'grnBlock', 'ornBlock', 'vioBlock', _NULL_NAME]


_POSE_DIM     = 7
_ACTUAL_NAMES = _BLOCK_NAMES[:-1]
_N_CLASSES    = len( _BLOCK_NAMES )
_BLOCK_SCALE  = 0.038
_MIN_X_OFFSET = 0.400



########## BELIEFS #################################################################################

_CONFUSE_PROB = 0.025
_NULL_THRESH  =  0.75
_EXIST_THRESH = 0.05



########## MEASUREMENTS ############################################################################

_ACCEPT_POSN_ERR =  1.00*_BLOCK_SCALE # 0.65 # 0.75 # 0.85
_Z_SAFE          =  8.0*_BLOCK_SCALE
_MIN_SEP         =  0.75*_BLOCK_SCALE # 0.40

