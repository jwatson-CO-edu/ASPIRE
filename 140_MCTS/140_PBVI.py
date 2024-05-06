########## INIT ####################################################################################

import numpy as np

MIN = -np.inf

########## HELPER STRUCTS ##########################################################################

class AlphaVec:
    """ An alpha vector """
    def __init__( self, a, v ):
        self.a = a
        self.v = v
    def copy( self ):
        return AlphaVec( self.a, self.v )
    


########## POINT BASED VALUE ITERATION SOLVER ######################################################
    
class PBVI:
    """ Point-Based Value Iteration """

    def __init__( self ):
        """ ??? """
        self.states   = []
        self.actions  = []
        self.observs  = []
        self.Tmodel   = None
        self.Rmodel   = None
        self.Omodel   = None
        self.belPnts  = None
        self.alphaVc  = []
        self.solved   = False
        self.Gamma_R  = {}
        self.discount = 0.95


    def init_belief_points( self, beliefPoints ):
        """ Create belief points and dummy alpha vectors """
        self.belPnts = list( beliefPoints ) # Belief points should be sampled from the initial distribution
        for _ in beliefPoints: 
            self.alphaVc.append(  AlphaVec( a = -1, v = np.zeros( len(self.states) ) )  )
        self.compute_Gamma_a_star()


    def compute_Gamma_a_star( self ):
        """ Is this supposed to be a matrix? """
        for a_i in self.actions:
            self.Gamma_R[ a_i ] = [self.Rmodel(a_i, s_j) for s_j in self.states]
            

    def compute_Gamma_a_o( self, a, o ):
        """ Update the set of alpha vectors, given an action and observation """
        # NOTE: `a` and `o` are indices
        Gamma_a_o = []
        for alpha in self.alphaVc:
            vec = np.zeros( len(self.states) ) # initialize the update vector [0, ... 0]
            for i, si in enumerate( self.states ):
                for j, sj in enumerate( self.states ):
                    vec[i] += self.discount * self.Tmodel(a, si, sj) * self.Omodel(a, sj, o) * alpha.v[j]
            Gamma_a_o.append( vec )


    def solve( self, T_hrzn ):
        """ Solve the POMDP up to `T_hrzn` """
        for _ in range( T_hrzn ):
            
            # First compute a set of updated vectors for every action/observation pair
            # Action(a) => Observation(o) => UpdateOfAlphaVector (a, o)
            Gamma_int = {}
            for a_i in self.actions:
                Gamma_int[ a_i ] = {}
                for o_j in self.observs:
                    Gamma_int[ a_i ][ o_j ] = self.compute_Gamma_a_o( a_i, o_j )

            # Now compute the cross sum
            Gamma_a_b = {}
            for a_j in self.actions:
                Gamma_a_b[a_j] = {}
                for i, b in enumerate( self.belPnts ):
                     Gamma_a_b[a_j][i] = self.Gamma_R[a_j].copy()
                     for o_k in self.observs:
                         best_alpha_idx = np.argmax(np.dot(Gamma_int[a_j][o_k], b))
                         Gamma_a_b[a_j][i] += Gamma_int[a_j][o_k][best_alpha_idx]

            # Finally compute the new(best) alpha vector set
            self.alphaVc = []
            valMax       = MIN
            for i, b in enumerate( self.belPnts ):
                bestAlpha, bestAction = None, None
                for a_j in self.actions:
                    val = np.dot(Gamma_a_b[a_j][i], b)
                    if bestAlpha is None or val > valMax:
                        valMax     = val
                        bestAlpha  = Gamma_a_b[a_j][i].copy()
                        bestAction = a_j
                self.alphaVc.append(  AlphaVec( a = bestAction, v = bestAlpha )  )
        self.solved = True


    def get_best_action( self, belief ):
        """ Return the optimal action, given `belief` """
        valMax    = MIN
        alphaBest = None
        for av in self.alphaVc:
            v = np.dot( av.v, belief )
            if v > valMax:
                valMax    = v
                alphaBest = av
        return alphaBest.a
    

    def update_belief( self, b, a, o ):
        """ Update current belief based on problem dynamics """
        bNew = []
        for s_j in self.states:
            Pr_o_prime = self.Omodel( a, s_j, o )
            total      = 0.0
            for i, s_i in enumerate( self.states ):
                total += self.Tmodel( a, s_i, s_j ) * float( b[i] )
            bNew.append( Pr_o_prime*total )
        # normalize
        bSum = sum( bNew )
        return [ x / bSum for x in bNew ]

    

