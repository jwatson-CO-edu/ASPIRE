import numpy as np
from trimesh import Trimesh

from geometry import origin_row_vec

########## COMPONENTS ##############################################################################

class Volume:
    """ Basic geometric representation """
    def __init__( self ):
        """ Geo Data """
        self.mesh = Trimesh()


class ObjectReading:
    """ Represents an object segmentation in space not yet integrated into the scene belief """
    def __init__( self, distribution = None, objPose = None ):
        self.dstr  = distribution if (distribution is not None) else {} # Current belief in each class
        self.pose  = objPose if isinstance(objPose, (list,np.ndarray)) else origin_row_vec() # Object pose
        self.label = ""