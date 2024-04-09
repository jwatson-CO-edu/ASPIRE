########## INIT ####################################################################################
import os
import numpy as np
import pybullet_data



########## PATHS ###################################################################################

ROBOT_URDF_PATH = os.path.expanduser( "~/MAGPIE/090_pdls_responsive/ur_e_description/urdf/ur5e.urdf" )
TABLE_URDF_PATH = os.path.join( pybullet_data.getDataPath(), "table/table.urdf" )



########## OBJECTS #################################################################################

_BLOCK_SCALE  = 0.038
_MIN_X_OFFSET = 0.400



########## BELIEFS #################################################################################

_CONFUSE_PROB = 0.025