########## INIT ####################################################################################
from itertools import count

import numpy as np

from utils import row_vec_to_pb_posn_ornt
from env_config import _ROUND_PLACES


########## SYMBOLS #################################################################################


class Object: 
    """ Model of any object we can go to in the world, real or not """
    num = count()

    def __init__( self, label, pose, ref = None  ):
        self.ref    = ref # - Belief from which this symbols was sampled
        self.label  = label # Sampled object label
        self.pose   = np.around( pose, _ROUND_PLACES ) #- Sampled object pose
        self.grasp  = None 
        self.config = None 
        self.action = None # Action to which this symbol was assigned
        self.index  = next( self.num )

    def posn_ornt( self ):
        """ Get the pose in PyBullet format """
        return row_vec_to_pb_posn_ornt( self.pose )
    
    def p_attached( self ):
        """ Return true if this symbol has been assigned to an action """
        return (self.ref is not None)

    def prob( self ):
        """ Get the current belief this symbol is true based on the belief this symbol was drawn from """
        if self.p_attached():
            return self.ref.labels[self.label]
        else:
            return 0.0
        
    def fresh( self ):
        """ Return the freshness of the ref where it exists, otherwise return False """
        if self.p_attached():
            return self.ref.fresh
        else:
            return False
        
    def detach( self ):
        """ Remove self from reference, then remove reference """
        if self.p_attached():
            self.ref.remove_symbol( self.index )
    
    def __repr__( self ):
        """ String representation, Including current symbol belief """
        return f"<{self.label} @ {self.pose}, P={self.prob() if (self.ref is not None) else None}>"
    
    
    
    
class Path:
    """ Trajectory """
    num = count()

    def __init__( self, wps = None ):
        """ Set init waypoints """
        self.wp    = wps if isinstance( wps, list ) else list()
        self.index = next( self.num )

    def add_wp( self, wp ):
        """ Add a waypoint """
        self.wp.append( wp )

    def __repr__( self ):
        """ Summary of path in terms of indices """
        return f"<Path {self.index}: x_{[x_i.index for x_i in self.wp]}>"