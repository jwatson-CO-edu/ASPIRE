########## INIT ####################################################################################
from itertools import count

import numpy as np

from utils import row_vec_to_pb_posn_ornt
from env_config import _ROUND_PLACES


########## SYMBOLS #################################################################################

class Pose:
    """ A pose in the lab frame """
    num = count()

    def __init__( self, val ):
        self.value = np.around( val, _ROUND_PLACES )
        self.index = next( self.num )

    def posn_ornt( self ):
        return row_vec_to_pb_posn_ornt( self.value )

    def __repr__( self ):
        return f"<Pose {self.index}: {self.value}>"


class Config:
    """ A robot configuration """
    num = count()

    def __init__( self, val ):
        self.value = np.around( val, _ROUND_PLACES )
        self.index = next( self.num )

    def __repr__( self ):
        return f"<Config {self.index}: {self.value}>"


class Object: 
    """ A named object, and its pose """
    num = count()
    def __init__( self, label, pose, ref = None  ):
        self.ref    = ref # - Belief from which this symbols was sampled
        self.label  = label # Sampled object label
        self.pose   = np.around( pose, _ROUND_PLACES ) #- Sampled object pose
        self.action = None #- Action to which this symbol was assigned
        self.index  = next( self.num )

    def prob( self ):
        """ Get the current belief this symbol is true based on the belief this symbol was drawn from """
        return self.ref.labels[self.label]
    
    def __repr__( self ):
        """ String representation, Including current symbol belief """
        return f"<{self.label} @ {self.pose}, P={self.prob() if (self.ref is not None) else None}>"
    
    def p_attached( self ):
        """ Return true if this symbol has been assigned to an action """
        return (self.action is not None)
    

class Grasp:
    """ Relates a target pose to an effector pose """
    num = count()
    def __init__( self, tgtPose, effPose ):
        self.tgtPose = tgtPose
        self.effPose = effPose
        self.index  = next( self.num )