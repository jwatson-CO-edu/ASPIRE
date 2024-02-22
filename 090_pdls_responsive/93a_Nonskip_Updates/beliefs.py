########## INIT ####################################################################################

import sys
from random import random
from pprint import pprint

import numpy as np
from scipy.stats import chi2

sys.path.append( "../" )

from utils import ( roll_outcome, get_confusion_matx, multiclass_Bayesian_belief_update, p_lst_has_nan, 
                    diff_norm, pb_posn_ornt_to_row_vec )

from env_config import ( _POSN_STDDEV, _BLOCK_NAMES, _NULL_NAME, _N_POSE_UPDT, _NEAR_PROB, _CONFUSE_PROB, 
                         _NULL_THRESH, _MIN_SEP, _EXIST_THRESH )

from symbols import Object



########## HYBRID BELIEFS ##########################################################################

class ObjectBelief:
    """ Hybrid belief: A discrete distribution of classes that may exist at a continuous distribution of poses """


    def reset_std_dev( self ):
        self.pStdDev = np.array([self.iStdDev for _ in range( 3 )]) 


    def __init__( self, initStddev = _POSN_STDDEV, nearThresh = _NEAR_PROB ):
        """ Initialize with origin poses and uniform, independent variance """
        self.labels  = {} # ---------------------- Current belief in each class
        self.mean    = np.array([0,0,0,]) # Mean pose
        self.ornt    = np.array([0,0,0,1,])
        self.iStdDev = initStddev
        self.reset_std_dev()# Position stddev
        self.pHist   = [] # ---------------------- Recent history of poses
        self.pThresh = nearThresh # --------------------- Minimum prob density at which a nearby pose is relevant
        self.visited = False
        self.fresh   = True
        self.symbols = [] 


    def spawn_object( self, label, pose ):
        """ Spawn a tracked object that references this belief """
        rtnObj = Object( label, pose, self )
        self.symbols.append( rtnObj )
        return rtnObj
    

    def index_of_symbol_index( self, idx ):
        """ Get the list index of the `Object.index`, else return -1 """
        for i, sym in enumerate( self.symbols ):
            if sym.index == idx:
                return i
        return -1


    def remove_symbol( self, idx ):
        """ Remove the symbol with the given `idx` """
        i = self.index_of_symbol_index( idx )
        if i >= 0:
            self.symbols[i].ref = None
            self.symbols.pop(i)


    def remove_all_symbols( self ):
        for sym in self.symbols:
            sym.ref = None
        self.symbols = [] 


    def posn( self ):
        """ Get the position """
        return np.array( self.mean[:3] )


    def posn_covar( self ):
        """ Get the pose covariance """
        rtnArr = np.zeros( (3,3,) )
        for i in range(3):
            rtnArr[i,i] = (self.pStdDev[i])**2
        return rtnArr


    def prob_density( self, obj ):
        """ Return the probability that this object lies within the present distribution """
        x     = np.array( obj.pose[:3] )
        mu    = self.posn()
        sigma = self.posn_covar()
        try:
            m_dist_x = np.dot((x-mu).transpose(),np.linalg.inv(sigma))
            m_dist_x = np.dot(m_dist_x, (x-mu))
            return 1-chi2.cdf( m_dist_x, 3 )
        except np.linalg.LinAlgError:
            return 0.0


    def p_reading_relevant( self, obj ):
        """ Roll die to determine if a nearby pose is relevant """
        return ( random() <= self.prob_density( obj ) )
    

    def sample_pose( self ):
        """ Sample a pose from the present distribution, Reset on failure """
        try:
            posnSample = np.random.multivariate_normal( self.posn(), self.posn_covar() ) 
        except (np.linalg.LinAlgError, RuntimeWarning,):
            self.reset_std_dev()
            posnSample = np.random.multivariate_normal( self.posn(), self.posn_covar() ) 
        while p_lst_has_nan( posnSample ):
            self.reset_std_dev()
            posnSample = np.random.multivariate_normal( self.posn(), self.posn_covar() ) 
        return pb_posn_ornt_to_row_vec( posnSample, self.ornt )


    def sample_symbol( self ):
        """ Sample a determinized symbol from the hybrid distribution """
        label = roll_outcome( self.labels )
        pose  = self.sample_pose()
        return self.spawn_object( label, pose )
    
    
    def sample_null( self ):
        """ Empty Pose """
        return self.spawn_object( _NULL_NAME, np.array( self.mean ) )
    
        
    def get_posn_history( self ):
        """ Get the positions of all pose history readings """
        hist = np.zeros( (len(self.pHist),3,) )
        for i, row in enumerate( self.pHist ):
            hist[i,:] = row[:3]
        return hist


    def update_pose_dist( self ):
        """ Update the pose distribution from the history of observations """
        self.fresh   = True
        posnHist     = self.get_posn_history()
        q_1_Hat      = np.array( self.mean )
        q_2_Hat      = np.mean( posnHist, axis = 0 )
        nuStdDev     = np.std(  posnHist, axis = 0 )
        omegaSqr_1   = np.dot( self.pStdDev, self.pStdDev )
        omegaSqr_2   = np.dot( nuStdDev    , nuStdDev     )
        self.pHist   = []
        try:
            self.mean    = q_1_Hat + omegaSqr_1 / ( omegaSqr_1 + omegaSqr_2 ).dot( q_2_Hat - q_1_Hat )
            self.pStdDev = np.sqrt( np.reciprocal( np.add(
                np.reciprocal( omegaSqr_1 ),
                np.reciprocal( omegaSqr_2 ),
            ) ) )
        except:
            print( "WARNING: Covariance reset due to overflow!" )
            self.mean = q_1_Hat
            self.reset_std_dev()

    
    def integrate_reading( self, objReading ):
        """ if `objBelief` is relevant, then Update this belief with evidence and return True, Otherwise return False """
        # NOTE: THIS WILL NOT BE AS CLEAN IF THE CLASSIFIER DOES NO PROVIDE A DIST ACROSS ALL CLASSES
        if self.p_reading_relevant( objReading ):
            Nclass = len( _BLOCK_NAMES )
            cnfMtx = get_confusion_matx( Nclass, _CONFUSE_PROB )
            priorB = [ self.labels[ label ] for label in _BLOCK_NAMES ] 
            evidnc = [ objReading.labels[ label ] for label in _BLOCK_NAMES ]
            updatB = multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc )
            self.labels = {}
            for i, name in enumerate( _BLOCK_NAMES ):
                self.labels[ name ] = updatB[i]
            self.pHist.append( objReading.pose )
            if len( self.pHist ) >= _N_POSE_UPDT:
                self.update_pose_dist()
            return True
        else:
            return False
        
    def integrate_null( self ):
        """ Accrue a non-observation """
        Nclass = len( _BLOCK_NAMES )
        labels = {}
        for i in range( len( _BLOCK_NAMES ) ):
            blkName_i = _BLOCK_NAMES[i]
            if blkName_i == _NULL_NAME:
                labels[ blkName_i ] = 1.0-_CONFUSE_PROB*(len( _BLOCK_NAMES )-1)
            else:
                labels[ blkName_i ] = _CONFUSE_PROB
        cnfMtx = get_confusion_matx( Nclass, _CONFUSE_PROB )
        priorB = [ self.labels[ label ] for label in _BLOCK_NAMES ] 
        evidnc = [ labels[ label ] for label in _BLOCK_NAMES ]
        updatB = multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc )
        self.labels = {}
        for i, name in enumerate( _BLOCK_NAMES ):
            self.labels[ name ] = updatB[i]



class ObjectMemory:
    """ Attempt to maintain recent and constistent object beliefs based on readings from the vision system """

    def __init__( self ):
        """ Set belief containers """
        self.beliefs = []
        self.last    = {}


    def reset_beliefs( self ):
        """ Remove all references to the beliefs, then erase the beliefs """
        for bel in self.beliefs:
            bel.remove_all_symbols()
        self.beliefs = []
        self.last    = {}


    def belief_from_reading( self, objReading ):
        """ Center a new belief on the incoming reading """
        nuBelief = ObjectBelief()
        nuBelief.labels = dict( objReading.labels )
        nuBelief.mean   = np.array( objReading.pose )
        return nuBelief


    def integrate_one_reading( self, objReading ):
        """ Fuse this belief with the current beliefs """
        # 1. Determine if this belief provides evidence for an existing belief
        relevant = False
        for belief in self.beliefs:
            if belief.integrate_reading( objReading ):
                relevant = True
                belief.visited = True
                # 2024-02-22: Let the dice decide how many beliefs this influences
                # Assume that this is the only relevant match, break
                # break
                # print( "NONSKIP!", end=", " )
        # 2. If this evidence does not support an existing belief, it is a new belief
        if not relevant:
            self.beliefs.append( self.belief_from_reading( objReading ) )
        return relevant


    def unvisit_beliefs( self ):
        """ Set visited flag to False for all beliefs """
        for belief in self.beliefs:
            belief.visited = False


    def erase_dead( self ):
        """ Erase all beliefs and cached symbols that no longer have relevancy """
        retain = []
        for belief in self.beliefs:
            if belief.labels[ _NULL_NAME ] < _NULL_THRESH:
                retain.append( belief )
            else:
                belief.remove_all_symbols()
                print( f"{str(belief)} DESTROYED!" )
        self.beliefs = retain
        delNam = []
        for k, v in self.last:
            if v.prob() < _EXIST_THRESH:
                delNam.append( k )
        for name in delNam:
            del self.last[ name ]


    def decay_beliefs( self ):
        """ Destroy beliefs that have accumulated too many negative indications """
        for belief in self.beliefs:
            if not belief.visited:
                belief.integrate_null()
        self.erase_dead()
        self.unvisit_beliefs()


    def belief_update( self, objEvidence ):
        """ Gather and aggregate evidence """

        ## Integrate Beliefs ##
        cNu = 0
        cIn = 0
        self.unvisit_beliefs()
        for objEv in objEvidence:
            if self.integrate_one_reading( objEv ):
                cIn += 1
            else:
                cNu += 1
        if (cNu or cIn):
            print( f"\t{cNu} new object beliefs this iteration!" )
            print( f"\t{cIn} object beliefs updated!" )
        else:
            print( f"\tNO belief update!" )
        
        self.decay_beliefs()
        
        print( f"Total Beliefs: {len(self.beliefs)}" )


    def scan_max_likelihood( self ):
        """ Get a list of unique samples, Keep only the most likely version of a label """
        smplSym = []
        for i in range(3):
            smplSym.extend( [bel.sample_symbol() for bel in self.beliefs] )
        uniqSym = {}
        for sym in smplSym:
            if sym.label != _NULL_NAME:
                if sym.label not in uniqSym:
                    uniqSym[ sym.label ] = sym
                elif sym.prob() > uniqSym[ sym.label ].prob():
                    uniqSym[ sym.label ] = sym
        return list( uniqSym.values() )


    def p_noncolliding( self, objs ):
        """ Return True if no two `objs` are within `_MIN_SEP` of each other """
        N = len( objs )
        for i, obj_i in enumerate( objs ):
            for obj_j in objs[(i+1):N]:
                if diff_norm( obj_i.pose[:3], obj_j.pose[:3] ) < _MIN_SEP:
                    return False
        return True


    def scan_consistent( self ):
        """ Get the most likely samples that do not collide with one another """
        uniqSym = self.scan_max_likelihood()
        while not self.p_noncolliding( uniqSym ):
            uniqSym = self.scan_max_likelihood()
        return uniqSym
    

    def set_beliefs_stale( self ):
        """ Set all beliefs not fresh """
        for bel in self.beliefs:
            bel.fresh = False


    def scan_consistent_fresh( self ):
        """ Keep only the most likely version of a label """
        smplSym = self.scan_consistent()
        # pprint( smplSym )
        frshNam = [sym.label for sym in smplSym if sym.fresh()]
        # pprint( frshNam )
        for sym in smplSym:
            if sym.label in frshNam:
                self.last[ sym.label ] = sym
        self.set_beliefs_stale()
        return list( self.last.values() )
    

    def sample_consistent( self, label ):
        """ Maintain a pose dict and only update it when needed """
        # self.scan_fresh()
        nuSym = self.scan_consistent()
        for sym in nuSym:
            if (sym.label == label):
                return sym
        return None
    

    def sample_consistent_fresh( self, label ):
        """ Maintain a pose dict and only update it when needed """
        self.scan_consistent_fresh()
        if label in self.last:
            return self.last[ label ]
        else:
            return None