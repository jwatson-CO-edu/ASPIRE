import numpy as np
from trimesh import Trimesh


from geometry import origin_row_vec, row_vec_to_homog

########## COMPONENTS ##############################################################################

class Volume:
    """ Basic geometric representation """

    def __init__( self ):
        """ Geo Data """
        self.mesh = Trimesh()

    
    def set_pose( self, rowVec ):
        """ Set the pose of the mesh """
        self.mesh.apply_transform( row_vec_to_homog( rowVec ) )


class ObjectReading:
    """ Represents an object segmentation in space not yet integrated into the scene belief """
    def __init__( self, distribution = None, objPose = None ):
        self.labels = distribution if (distribution is not None) else {} # Current belief in each class
        self.pose   = objPose if isinstance(objPose, (list,np.ndarray)) else origin_row_vec() # Object pose
        self.label  = ""