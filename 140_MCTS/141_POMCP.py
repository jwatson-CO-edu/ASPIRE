########## INIT ####################################################################################
import time
from time import sleep
now = time.time


########## NODE ####################################################################################

class TNode:
    """ Tree node for POMCP """
    def __init__( self ):
        """ Init empty node """
        self.s  = tuple()
        self.N  = 0
        self.V  = 0.0
        self.B  = list()
        self.ch = list()



########## POMCP ###################################################################################

class POMCP:
    """ Solver for large POMDPs """

    def __init__( self, hist = None ):
        """ Init empty history """
        self.h    = hist
        self.Tbgn = 0

    def Search( self, Timeout_s = 30.0 ):
        """ Search the history tree """
        self.Tbgn = now()
        while( (now()-self.Tbgn) < Timeout_s ):
            if self.h is None:
                pass

        
""" The reason I am leaving this behind is that POMCP uses a search tree to perform a particle filter.
    The method I have chosen is already a Bayes' filter.
    Also, POMCP updates the particle filter between actions as part of simulation.
    My method updates the Bayes filter during actions based on real data, not simulation. """