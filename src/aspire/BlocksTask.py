########## INIT ####################################################################################
import os
from random import random

import numpy as np

from env_config import ( _BLOCK_SCALE, _MIN_X_OFFSET, _MIN_Y_OFFSET, _X_WRK_SPAN, _Y_WRK_SPAN,)
from magpie.poses import repair_pose


_poseGrn = np.eye(4)
_poseGrn[0:3,3] = [ _MIN_X_OFFSET+_X_WRK_SPAN/2.0, _MIN_Y_OFFSET+_Y_WRK_SPAN/2.0, 0.5*_BLOCK_SCALE, ]

from symbols import ObjPose, extract_pose_as_homog, euclidean_distance_between_symbols


########## ENVIRONMENT #############################################################################

def set_block_env():
    os.environ["_trgtGrn"] = ObjPose( _poseGrn )


    os.environ["_temp_home"] = np.array( [[-1.000e+00, -1.190e-04,  2.634e-05, -2.540e-01],
                            [-1.190e-04,  1.000e+00, -9.598e-06, -4.811e-01],
                            [-2.634e-05, -9.601e-06, -1.000e+00,  4.022e-01],
                            [ 0.000e+00,  0.000e+00,  0.000e+00,  1.000e+00],] )

    os.environ["_GOOD_VIEW_POSE"] = repair_pose( np.array( [[-0.749, -0.513,  0.419, -0.428,],
                                            [-0.663,  0.591, -0.46 , -0.273,],
                                            [-0.012, -0.622, -0.783,  0.337,],
                                            [ 0.   ,  0.   ,  0.   ,  1.   ,],] ) )

    os.environ["_HIGH_VIEW_POSE"] = repair_pose( np.array( [[-0.709, -0.455,  0.539, -0.51 ],
                                            [-0.705,  0.442, -0.554, -0.194],
                                            [ 0.014, -0.773, -0.635,  0.332],
                                            [ 0.   ,  0.   ,  0.   ,  1.   ],] ) )

    os.environ["_HIGH_TWO_POSE"] = repair_pose( np.array( [[-0.351, -0.552,  0.756, -0.552],
                                            [-0.936,  0.194, -0.293, -0.372],
                                            [ 0.015, -0.811, -0.585,  0.283],
                                            [ 0.   ,  0.   ,  0.   ,  1.   ],] ) )


########## HELPER FUNCTIONS ########################################################################


def rand_table_pose():
    """ Return a random pose in the direct viscinity if the robot """
    rtnPose = np.eye(4)
    rtnPose[0:3,3] = [ 
        os.environ["_MIN_X_OFFSET"] + 0.5*os.environ["_X_WRK_SPAN"] + 0.5*os.environ["_X_WRK_SPAN"]*random(), 
        os.environ["_MIN_Y_OFFSET"] + 0.5*os.environ["_Y_WRK_SPAN"] + 0.5*os.environ["_Y_WRK_SPAN"]*random(), 
        os.environ["_BLOCK_SCALE"]/2.0,
    ]
    return rtnPose



########## Stream Creators #########################################################################

class BlockStreams:

    def __init__( self, planner ):
        """ Attach to Planner """
        self.planner = planner

    def get_above_pose_stream( self ):
        """ Return a function that returns poses """

        def stream_func( *args ):
            """ A function that returns poses """

            if os.environ["_VERBOSE"]:
                print( f"\nEvaluate ABOVE LABEL stream with args: {args}\n" )

            objcName = args[0]

            for sym in self.planner.symbols:
                if sym.label == objcName:
                    upPose = extract_pose_as_homog( sym )
                    upPose[2,3] += os.environ["_BLOCK_SCALE"]

                    rtnPose = self.planner.get_grounded_fact_pose_or_new( upPose )
                    print( f"FOUND a pose {rtnPose} supported by {objcName}!" )

                    yield (rtnPose,)

        return stream_func


    def get_placement_pose_stream( self ):
        """ Return a function that returns poses """

        def stream_func( *args ):
            """ A function that returns poses """

            if os.environ["_VERBOSE"]:
                print( f"\nEvaluate PLACEMENT POSE stream with args: {args}\n" )

            placed   = False
            testPose = None
            while not placed:
                testPose  = rand_table_pose()
                print( f"\t\tSample: {testPose}" )
                for sym in self.planner.symbols:
                    if euclidean_distance_between_symbols( testPose, sym ) < ( os.environ["_MIN_SEP"] ):
                        collide = True
                        break
                if not collide:
                    placed = True
            yield (ObjPose(testPose),)

        return stream_func


    def get_free_placement_test( self ):
        """ Return a function that checks placement poses """

        def test_func( *args ):
            """ a function that checks placement poses """
            print( f"\nEvaluate PLACEMENT test with args: {args}\n" )
            # ?pose
            chkPose   = args[0]
            print( f"Symbols: {self.planner.symbols}" )

            for sym in self.planner.symbols:
                if euclidean_distance_between_symbols( chkPose, sym ) < ( os.environ["_MIN_SEP"] ):
                    print( f"PLACEMENT test FAILURE\n" )
                    return False
            print( f"PLACEMENT test SUCCESS\n" )
            return True
        
        return test_func