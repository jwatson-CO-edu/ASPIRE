########## INIT ####################################################################################
import numpy as np

from env_config import _NULL_NAME



########## COMPONENTS ##############################################################################

class GraspObj:
    """ The concept of a named object at a pose """
    def __init__( self, label = None, pose = None ):
        self.label = label if (label is not None) else _NULL_NAME
        self.pose  = pose if isinstance(pose, (list,np.ndarray)) else np.array([0,0,0,1,0,0,0])