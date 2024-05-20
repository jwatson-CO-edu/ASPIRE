""" ##### POMCP #####
* Each node of the search tree estimates the value of a history by Monte-Carlo simulation.
* A history is a sequence of actions and observations, h_t = {a1, o1, ..., at, ot}
* Each particle corresponds to a sample state, and the belief state is the sum of all particles
* policy π(h, a) maps a history h to a probability distribution over actions
* After each simulation, one new node is added to the search tree
"""

########## INIT ####################################################################################
import time
from time import sleep
now = time.time
from random import random, randrange, choice

import numpy as np



########## PROBLEM DOMAIN ##########################################################################

class DomainModel:
    """ Model of the problem domain """
    def __init__( self ):
        self.A = list()
    def G( self, s, a ):
        raise NotImplementedError( "There is NO generative model!" )
    

class RockSample:
    """ POMDP RockSample Problem """

    def gen_rocks( self, k, goodProb ):
        """ Generate `k` rocks at random locations with a `goodProb` chance of being good """
        self.valID = [None for _ in range(k)]
        for i in range( k ):
            val = 1 if (random() <= goodProb) else 0
            adr = (randrange(0,self.xMax), randrange(0,self.yMax),)
            while adr in self.rocks:
                adr = (randrange(0,self.xMax), randrange(0,self.yMax),)
            self.rocks[ adr ] = val
            self.valID[i] = (adr, val,)
            

    def __init__( self, length, width, k, goodProb = 0.5 ):
        """ Set params and generate rocks """
        self.xMax  = width
        self.yMax  = length
        self.rocks = dict()
        self.gen_rocks( k, goodProb )
        self.A = [f"Chk{R}" for R in range(k)]
        self.A.extend( ['North','South','East','West','Sample'] )
        self.state = (randrange(0,self.xMax), randrange(0,self.yMax),)


    def sample_s_init_dist( self ):
        """ Sample a state from a uniformly random initial distribution """
        rtnStt = [self.state,]
        rtnStt.extend( [randrange(0,2) for _ in range( len(self.valID) )] )
        return rtnStt



    def G( self, s, a ):
        """ Outcome from generative model """
        sP = s
        o  = None
        r  = 0
        if a == 'Sample':
            if s in self.rocks:
                if self.rocks[s] == 1:
                    r = +10
                    o = 1
                    self.rocks[s] = 0
                else:
                    r = -10
                    o = 0
        elif a == 'North':
            if s[1] > 0:
                sP = (s[0],s[1]-1,)
        elif a == 'South':
            if s[1] < self.yMax-1:
                sP = (s[0],s[1]+1,)
        elif a == 'East':
            if s[0] < self.yMax-1:
                sP = (s[0]+1,s[1],)
            else:
                r = 10
        elif a == 'West':
            if s[0] > 0:
                sP = (s[0]-1,s[1],)
        elif 'Chk' in a:
            rkID = int( a[3:] )
            dist = np.linalg.norm( np.array(self.valID[ rkID ][0]) - np.array(s))
            if random() <= 0.5 + 1.0/(2.0 + dist**2):
                o = self.valID[ rkID ][1]
            else:
                o = (self.valID[ rkID ][1]+1)%2
        else:
            raise ValueError( "THIS SHOULD NOT HAVE HAPPENED!" )
        return (sP, o, r)
                
            
########## NODE ####################################################################################

class TNode:
    """ Tree node for POMCP """

    def __init__( self ):
        """ Init empty node """
        # Structure #
        self.tp = "ROOT" # Type
        self.pa = None # - Parent 
        self.ch = list() # Children
        # Contents #
        self.ao = None # -- Action / Observation
        self.N  = 0 # ----- Number of times this history has been visited
        self.V  = 0.0 # --- Value of this history


    def p_contains( self, h, idx = 0 ):
        """ Return `True` if the subtree contains the history, Otherwise return `False` """
        if h[ idx ] == self.ao:
            if idx >= (len(h) - 1):
                return True
            for child in self.ch:
                if child.p_contains( h, idx+1 ):
                    return True
        return False
    

    def add_child( self, nuNode ):
        """ Add a parent-child relationship between this node and `nuNode` """
        nuNode.pa = self
        self.ch.append( nuNode )



def make_A_node( action = None ):
    rtNode = TNode()
    rtNode.tp = "act"
    rtNode.ao = action
    return rtNode


def make_O_node( observ = None ):
    rtNode = TNode()
    rtNode.tp = "obs"
    rtNode.ao = observ
    return rtNode




########## POMCP ###################################################################################

class POMCP:
    """ Solver for large, discrete POMDPs """

    def __init__( self, probModel = None, hist = None ):
        """ Init empty history """
        self.T     = TNode() # - Tree
        self.tLst  = 0 # ------- Start time for anytime portion
        self.B     = [] # ------ Belief
        self.epsln = 1e-4 # ---- Epsilon stopping condition
        self.gamma = 0.99 # ---- Discount
        self.model = probModel # Transition Model
        self.piRO  = None # ---- Rollout Policy
        if probModel is None:
            self.model = RockSample( 11, 11, 10, 0.5 )


    def Sample_I( self ):
        """ Sample from the initial distribution """
        return self.model.sample_s_init_dist()


    def Sample_B( self ):
        """ Sample from the current belief particles """
        return choice( self.B )


    def pi_action_sel_RO( self, h ):
        """ Use the Rollout Policy to select an action that is estimated optimal for the given history """
        # Choose an action randomly from history?
        # FIXME: HISTORY IS A TREE, NOT A LIST
        a = choice( h )
        while a not in self.model.A:
            a = choice( h )
        return a
    

    def Rollout( self, s, h, d ):
        """ MC from state to end --> discounted reward """
        if self.gamma**d < self.epsln:
            return 0
        a = self.pi_action_sel_RO( h )
        (sP, o, r) = self.model.G( s, a )
        hP = list( h ).extend( [a, o] )
        return r + self.gamma * self.Rollout( sP, hP, d+1 )


    def Simulate( self, s, h, d ):
        """ Traverse tree and extend with estimated value(s) """
        if self.gamma**d < self.epsln:
            return 0
        if not self.T.p_contains( h ):
            for a in self.model.A:
                # FIXME: WHERE THE HECK DO I PUT THESE NODES?
                pass
            self.Rollout( s, h, d )


    def Search( self, Timeout_s = 30.0 ):
        """ Search the history tree """
        self.tLst = now()
        while( (now()-self.tLst) < Timeout_s ):
            if self.h is None:
                s = self.Sample_I()
            else:
                s = self.Sample_B()
            self.Simulate( s, self.h, 0 )


    def Belief_Update( self, a, o ):
        """ Update the belief particle filter with the actual (a,o) """
        # FIXME: DESCRIBED BUT NO ALGO
        pass

        
