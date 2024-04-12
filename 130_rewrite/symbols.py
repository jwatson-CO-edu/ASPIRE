########## INIT ####################################################################################
from copy import deepcopy
from itertools import count

import numpy as np

from env_config import _NULL_NAME



########## COMPONENTS ##############################################################################

class GraspObj:
    """ The concept of a named object at a pose """
    num = count()

    def __init__( self, label = None, pose = None ):
        self.label = label if (label is not None) else _NULL_NAME
        self.pose  = pose if isinstance(pose, (list,np.ndarray)) else np.array([0,0,0,1,0,0,0])
        self.index = next( self.num )


    def __repr__( self ):
        """ Text representation of noisy reading """
        return f"<GraspObj@{self.pose}, Class: {str(self.label)}>"



class ObjectReading( GraspObj ):
    """ Represents a signal coming from the vision pipeline """

    def __init__( self, labels = None, pose = None ):
        """ Init distribution and  """
        super().__init__( None, pose )
        self.labels  = labels if (labels is not None) else {} # Current belief in each class
        self.visited = False


    def __repr__( self ):
        """ Text representation of noisy reading """
        return f"<GraspObj@{self.pose}, Class: {str(self.labels)}>"
    

    def copy( self ):
        """ Make a copy of this belief """
        return ObjectReading( deepcopy( self.labels ), np.array( self.pose ) )
