import trimesh
from trimesh import Trimesh


class Volume:
    """ Basic geometric representation """

    def __init__( self ):
        """ Geo Data """
        self.mesh = Trimesh()


class Object:
    """ A physical thing that the robot has beliefs about """

    def __init__( self, pose = None, volume = None ):
        self.muPose = pose if (pose is not None) else [0,0,0,1,0,0,0]
        self.volume = volume if (volume is not None) else Volume()
        self.stddev = 
