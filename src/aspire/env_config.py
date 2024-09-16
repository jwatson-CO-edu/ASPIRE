########## INIT ####################################################################################
import os
import numpy as np
# import pybullet_data



########## SETTINGS ################################################################################

def set_global_env():
    np.set_printoptions(
        edgeitems =  16, # Number of items before ...
        linewidth = 200, 
        formatter = dict( float = lambda x: "%.5g" % x ) 
    )

    # ROBOT_URDF_PATH = os.path.expanduser( "~/MAGPIE/090_pdls_responsive/ur_e_description/urdf/ur5e.urdf" )
    # TABLE_URDF_PATH = os.path.join( pybullet_data.getDataPath(), "table/table.urdf" )
    os.environ["_VERBOSE"] = True



########## GRAPHICS ################################################################################

def set_grahpics_env():
    os.environ["_USE_GRAPHICS"]   = False
    os.environ["_RECORD_SYM_SEQ"] = True
    os.environ["_BLOCK_ALPHA"]  = 1.0
    os.environ["_CLR_TABLE"]  = {
        'red': [1.0, 0.0, 0.0,],
        'ylw': [1.0, 1.0, 0.0,],
        'blu': [0.0, 0.0, 1.0,],
        'grn': [0.0, 1.0, 0.0,],
        'orn': [1.0, 0.5, 0.0,],
        'vio': [0.5, 0.0, 1.0,]
    }



########## OBJECTS #################################################################################

def set_object_env():
    os.environ["_NULL_NAME"]       = "NOTHING"

    os.environ["_ONLY_RED"]        = False
    os.environ["_ONLY_PRIMARY"]    = False
    os.environ["_ONLY_SECONDARY"]  = False
    os.environ["_ONLY_EXPERIMENT"] = True

    if os.environ["_ONLY_RED"]:
        os.environ["_BLOCK_NAMES"]  = ['redBlock', os.environ["_NULL_NAME"]]
    elif os.environ["_ONLY_PRIMARY"]:
        _BLOCK_NAMES  = ['redBlock', 'ylwBlock', 'bluBlock', os.environ["_NULL_NAME"],]
    elif os.environ["_ONLY_SECONDARY"]:
        _BLOCK_NAMES  = ['grnBlock', 'ornBlock', 'vioBlock', os.environ["_NULL_NAME"],]
    elif os.environ["_ONLY_EXPERIMENT"]:
        _BLOCK_NAMES  = ['grnBlock', 'bluBlock', 'ylwBlock', os.environ["_NULL_NAME"],]
    else:
        _BLOCK_NAMES  = ['redBlock', 'ylwBlock', 'bluBlock', 'grnBlock', 'ornBlock', 'vioBlock', os.environ["_NULL_NAME"],]


    os.environ["_POSE_DIM"]     = 7
    os.environ["_ACTUAL_NAMES"] = os.environ["_BLOCK_NAMES"][:-1]
    os.environ["_N_CLASSES"]    = len( os.environ["_BLOCK_NAMES"]  )
    os.environ["_N_ACTUAL"]     = len( os.environ["_ACTUAL_NAMES"] )

    os.environ["_BLOCK_SCALE"]  = 0.025 # Medium Wooden Blocks (YCB)
    # _BLOCK_SCALE  = 0.040 # 3D Printed Blocks

    os.environ["_ACCEPT_POSN_ERR"] = 0.55*os.environ["_BLOCK_SCALE"] # 0.50 # 0.65 # 0.75 # 0.85 # 1.00
    os.environ["_Z_SAFE"]          = 0.400
    os.environ["_MIN_SEP"]         = 0.85*os.environ["_BLOCK_SCALE"] # 0.40 # 0.60 # 0.70 # 0.75

    os.environ["_SPACE_EXPAND"] =  0.100
    os.environ["_MIN_X_OFFSET"] = -0.380 - os.environ["_SPACE_EXPAND"]
    os.environ["_MAX_X_OFFSET"] = -0.060 + os.environ["_SPACE_EXPAND"]
    os.environ["_MIN_Y_OFFSET"] = -0.614 - os.environ["_SPACE_EXPAND"]
    os.environ["_MAX_Y_OFFSET"] = -0.290 + os.environ["_SPACE_EXPAND"]
    os.environ["_MAX_Z_BOUND"]  = os.environ["_BLOCK_SCALE"]*4.0
    os.environ["_X_WRK_SPAN"] = os.environ["_MAX_X_OFFSET"] - os.environ["_MIN_X_OFFSET"]
    os.environ["_Y_WRK_SPAN"] = os.environ["_MAX_Y_OFFSET"] - os.environ["_MIN_Y_OFFSET"]



########## ROBOT ###################################################################################

def set_robot_env():
    os.environ["_ROBOT_FREE_SPEED"]   =  0.125
    os.environ["_ROBOT_HOLD_SPEED"]   =  0.125
    os.environ["_MOVE_COOLDOWN_S"]    =  0.5
    os.environ["_BT_UPDATE_HZ"]       = 10.0
    os.environ["_BT_ACT_TIMEOUT_S"]   = 20.0




########## CAMERA ##################################################################################

def set_camera_env():
    os.environ["_D405_FOV_H_DEG"]     = 84.0
    os.environ["_D405_FOV_V_DEG"]     = 58.0
    os.environ["_D405_FOV_D_M"]       =  0.920
    os.environ["_MIN_CAM_PCD_DIST_M"] = 0.075 + os.environ["_BLOCK_SCALE"] * len( os.environ["_ACTUAL_NAMES"] )


