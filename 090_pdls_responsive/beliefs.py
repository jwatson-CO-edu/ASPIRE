########## INIT ####################################################################################

import numpy as np
from scipy.stats import chi2

from utils import ( roll_outcome, get_confusion_matx, multiclass_Bayesian_belief_update, p_lst_has_nan, 
                    row_vec_to_pb_posn_ornt, pb_posn_ornt_to_row_vec )

from env_config import ( _POSN_STDDEV, _BLOCK_NAMES, _NULL_NAME, _N_POSE_UPDT, _NEAR_PROB, _CONFUSE_PROB )

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
        """ Determine if a nearby pose is relevant """
        return ( self.prob_density( obj ) >= self.pThresh )
    

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
        return Object( 
            label, 
            pose,
            self
        )
    
    def sample_null( self ):
        """ Empty Pose """
        return Object( 
            _NULL_NAME, 
            [0,0,0,1,0,0,0],
            self
        )
    
    def sample_fresh( self ):
        """ Only return a labeled pose if it is fresh, That is sampled after a pose update """
        if self.fresh:
            self.fresh = False
            return self.sample_symbol()
        else: 
            return self.sample_null()
        
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
        if self.p_pose_relevant( objReading ):
            Nclass = len( _BLOCK_NAMES )
            cnfMtx = get_confusion_matx( Nclass, _CONFUSE_PROB )
            priorB = [ self.labels[ label ] for label in _BLOCK_NAMES ] 
            evidnc = [ objReading.dstr[ label ] for label in _BLOCK_NAMES ]
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


class ObjectMemory:
    # FIXME, START HERE: MOVE ALL SCANS AND CONSISTENCY CHECKS HERE
    pass