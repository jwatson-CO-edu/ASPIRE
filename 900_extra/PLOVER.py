########## INIT ####################################################################################

##### Imports #####

### Standard ###
from uuid import uuid5
from random import random

### Special ###
import numpy as np

from scipy.stats import chi2

### Local ###
from env_config import _POSN_STDDEV, _ORNT_STDDEV, _NULL_NAME, _CONFUSE_PROB
from utils import ( p_lst_has_nan, roll_outcome, row_vec_normd_ornt, get_confusion_matx,
                    multiclass_Bayesian_belief_update )



########## SCENE GRAPH #############################################################################


class SpatialNode:
    """ A concept that can be situated in space and participate in relationships """
    # NOTE: This can also be used as a non-physical reference frame

    def __init__( self, label = "", pose = None, volume = None ):
        self.ID       = uuid5() # --------------------------------------- Means for identifying an unique object
        self.label    = label # ----------------------------------------- Text label, possibly non-unique
        self.pose     = pose if (pose is not None) else [0,0,0,1,0,0,0] # Absolute pose
        self.relPose  = [0,0,0,1,0,0,0] # ------------------------------- Relative pose
        self.volume   = volume if (volume is not None) else Volume() # -- Defining volume in space
        self.data     = {} # -------------------------------------------- TBD
        self.incoming = {} # -------------------------------------------- Upstream 
        self.outgoing = {} # -------------------------------------------- Downstream
        # TDB: Give nodes a lifespan so that we avoid collisions with them when sequencing actions?
        # BEGIN TIME?
        # END TIME?
        


class Object( SpatialNode ):
    """ A determinized instance of an object belief """

    def __init__( self, label = "", pose = None, volume = None, ref = None ):
        """ Set pose Gaussian and geo info """
        super().__init__( label, pose, volume )
        self.ref = ref



class ObjectBelief( SpatialNode ):
    """ The concept of a physical thing that the robot has beliefs about """

    ##### Init ############################################################

    def reset_pose_distrib( self ):
        """ Reset the pose distribution """
        self.stddev = [_POSN_STDDEV for _ in range(3)] # Standard deviation of pose
        self.stddev.extend( [_ORNT_STDDEV for _ in range(4)] )


    def __init__( self, label = "", pose = None, volume = None ):
        """ Set pose Gaussian and geo info """
        super().__init__( label, pose, volume )
        self.symbols = {} # All symbols sampled from this object concept
        self.labels  = {} # Discrete dist over labels for this object
        self.pHist   = [] # Recent history of poses
        self.reset_pose_distrib()


    ##### Symbol Memory ###################################################

    def spawn_symbol( self, label, pose ):
        """ Spawn a tracked object that references this belief """
        rtnObj = Object( label, pose, self.volume, self )
        self.symbols[ rtnObj.ID ] = rtnObj
        return rtnObj
    

    def remove_symbol( self, sym ):
        """ Remove the symbol with the given `idx` """
        if sym.ID in self.symbols:
            sym.ref = None
            del self.symbols[ sym.ID ]


    def remove_all_symbols( self ):
        for sym in self.symbols.values():
            sym.ref = None
        self.symbols = {}


    ##### Probability & Sampling ##########################################
        
    def pose_covar( self ):
        """ Get the pose covariance """
        rtnArr = np.zeros( (7,7,) )
        for i in range(7):
            rtnArr[i,i] = (self.stddev[i])**2
        return rtnArr
    

    def prob_density( self, obj ):
        """ Return the probability that this object lies within the present distribution """
        x     = np.array( obj.pose )
        mu    = np.array( self.pose )
        sigma = self.pose_covar()
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
            posnSample = np.random.multivariate_normal( self.pose, self.pose_covar() ) 
        except (np.linalg.LinAlgError, RuntimeWarning,):
            self.reset_pose_distrib()
            posnSample = np.random.multivariate_normal( self.pose, self.pose_covar() ) 
        while p_lst_has_nan( posnSample ):
            self.reset_std_dev()
            posnSample = np.random.multivariate_normal( self.pose, self.pose_covar() ) 
        return row_vec_normd_ornt( posnSample )
    

    def sample_symbol( self ):
        """ Sample a determinized symbol from the hybrid distribution """
        label = roll_outcome( self.labels )
        pose  = self.sample_pose()
        return self.spawn_symbol( label, pose )
    

    def sample_null( self ):
        """ Empty Pose """
        return self.spawn_symbol( _NULL_NAME, np.array( self.pose ) )
    

    def get_pose_history( self ):
        """ Get all pose history readings as a 2D matrix where each row is a reading """
        hist = np.zeros( (len(self.pHist),7,) )
        for i, row in enumerate( self.pHist ):
            hist[i,:] = row
        return hist
    

    def update_pose_dist( self ):
        """ Update the pose distribution from the history of observations """
        self.fresh   = True
        poseHist     = self.get_pose_history()
        q_1_Hat      = np.array( self.mean )
        q_2_Hat      = np.mean( poseHist, axis = 0 )
        nuStdDev     = np.std(  poseHist, axis = 0 )
        omegaSqr_1   = np.dot( self.stddev, self.stddev )
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
            self.reset_pose_distrib()


    def sorted_labels( self ):
        """ Get the label dist keys in a PREDICTABLE ORDER """
        rtnLst = list( self.labels.keys() )
        rtnLst.sort()
        return rtnLst


    def integrate_reading( self, objReading ):
        """ if `objReading` is relevant, then Update this belief with evidence and return True, Otherwise return False """
        # FIXME: HANDLE THE CASE WHERE THE READING HAS *DIFFERENT* CLASSES IN ITS DISTRIBUTION THAN THIS BELIEF,
        #        IN THIS CASE, INCOMING EVIDENCE FOR NON-REPRESENTED CLASSES SHOULD BE *ZERO*
        raise NotImplementedError( "`integrate_reading` does NOT account for unseen labels!" )
    

    def integrate_null( self ):
        """ Accrue a non-observation """
        Nclass  = len( self.labels )
        labels  = {}
        
        if _NULL_NAME not in self.labels:
            self.labels[ _NULL_NAME ] = 0.0

        ordLbls = self.sorted_labels()

        for objName_i in ordLbls:
            if objName_i == _NULL_NAME:
                # FIXME: `_CONFUSE_PROB` SHOULD NOT BE A CONSTANT THING AND COMES FROM THE SPECIFIC CLASSIFIER
                labels[ objName_i ] = 1.0-_CONFUSE_PROB*(Nclass-1)
            else:
                labels[ objName_i ] = _CONFUSE_PROB
        cnfMtx = get_confusion_matx( Nclass, _CONFUSE_PROB )
        priorB = [ self.labels[ label ] for label in ordLbls ] 
        evidnc = [ labels[ label ] for label in ordLbls ]
        updatB = multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc )
        self.labels = {}
        for i, name in enumerate( ordLbls ):
            self.labels[ name ] = updatB[i]
