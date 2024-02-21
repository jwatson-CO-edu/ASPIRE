########## INIT ####################################################################################
from uuid import uuid5

from trimesh import Trimesh

from env_config import _POSN_STDDEV, _ORNT_STDDEV



########## COMPONENTS ##############################################################################


class Volume:
    """ Basic geometric representation """

    def __init__( self ):
        """ Geo Data """
        self.mesh = Trimesh()



########## SCENE GRAPH #############################################################################


class SpatialNode:
    """ A concept that can be situated in space and participate in relationships """

    def __init__( self, label = "", pose = None ):
        self.ID       = uuid5()
        self.label    = label
        self.pose     = pose if (pose is not None) else [0,0,0,1,0,0,0]
        self.data     = {}
        self.incoming = {} # Upstream
        self.outgoing = {} # Downstream
        # BEGIN TIME?
        # END TIME?
        


class Object( SpatialNode ):
    """ A physical thing that the robot has beliefs about """

    def reset_pose_distrib( self ):
        """ Reset the pose distribution """
        self.stddev = [_POSN_STDDEV for _ in range(3)]
        self.stddev.extend( [_ORNT_STDDEV for _ in range(4)] )

    def __init__( self, label = "", pose = None, volume = None ):
        """ Set pose Gaussian and geo info """
        super().__init__( label, pose )
        self.volume = volume if (volume is not None) else Volume()
        self.reset_pose_distrib()
        
