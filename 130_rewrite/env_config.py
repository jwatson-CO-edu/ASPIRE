########## INIT ####################################################################################
import os
import numpy as np
import pybullet_data



########## PATHS ###################################################################################

ROBOT_URDF_PATH = os.path.expanduser( "~/MAGPIE/090_pdls_responsive/ur_e_description/urdf/ur5e.urdf" )
TABLE_URDF_PATH = os.path.join( pybullet_data.getDataPath(), "table/table.urdf" )



########## PYBULLET ################################################################################

_USE_GRAPHICS = False
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
_BLOCK_SCALE  = 0.038
_MIN_X_OFFSET = 0.400



########## BELIEFS #################################################################################

_CONFUSE_PROB = 0.025