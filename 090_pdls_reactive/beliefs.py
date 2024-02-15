########## INIT ####################################################################################

import numpy as np
from scipy.stats import chi2

from utils import ( roll_outcome, get_confusion_matx, multiclass_Bayesian_belief_update  )

from env_config import ( _SUPPORT_NAME, _POSN_STDDEV, _BLOCK_NAMES, _NULL_NAME, _N_POSE_UPDT, _POSE_DIM )

from symbols import Object



########## SYMBOLS & BELIEFS #######################################################################


class ObjectBelief:
    """ Hybrid belief: A discrete distribution of classes that may exist at a continuous distribution of poses """

    def reset_covar( self ):
        self.covar   = np.zeros( (_POSE_DIM,_POSE_DIM,) ) # ------ Pose covariance matrix
        for i, stdDev in enumerate( self.pStdDev ):
            self.covar[i,i] = stdDev * stdDev

    def __init__( self, initStddev = _POSN_STDDEV ):
        """ Initialize with origin poses and uniform, independent variance """
        # stdDev = [initStddev if (i<3) else 0.0 for i in range(7)]
        stdDev = [initStddev for i in range( _POSE_DIM )]
        self.labels  = {} # ---------------------- Current belief in each class
        self.pose    = np.array([0,0,0,1,0,0,0]) # Mean pose
        self.pStdDev = np.array(stdDev) # -------- Pose variance
        self.pHist   = [] # ---------------------- Recent history of poses
        self.pThresh = 0.5 # --------------------- Minimum prob density at which a nearby pose is relevant
        self.reset_covar()
        self.visited = False
        self.fresh   = True

    def get_posn( self, poseOrBelief ):
        """ Get the position from the object """
        if isinstance( poseOrBelief, (ObjectBelief, Object) ):
            return poseOrBelief.pose[0:3]
        elif isinstance( poseOrBelief, np.ndarray ):
            if poseOrBelief.size == (4,4,):
                return poseOrBelief[0:3,3]
            else:
                return poseOrBelief[0:3]

    def p_pose_relevant( self, poseOrBelief ):
        """ Determine if a nearby pose is relevant """
        x        = self.get_posn( poseOrBelief )
        mu       = self.get_posn( self.pose )
        sigma    = self.covar[0:3,0:3]
        m_dist_x = np.dot((x-mu).transpose(),np.linalg.inv(sigma))
        m_dist_x = np.dot(m_dist_x, (x-mu))
        return (1-chi2.cdf( m_dist_x, 3 ) >= self.pThresh)
    
    def object_supporting_pose( self, pose ):
        """ Return the name of the object that this pose is resting on """
        # GRIEF: GEO GRAMMAR WOULD BE GOOD HERE
        # HACK:  WE ONLY SAMPLE RAW POSES FOR UNSTACKED BLOCKS, SO ALWAYS RETURN TABLE
        return _SUPPORT_NAME

    def sample_symbol( self ):
        """ Sample a determinized symbol from the hybrid distribution """
        label = roll_outcome( self.labels )
        try:
            poseSample = np.random.multivariate_normal( self.pose, self.covar ) 
        except (np.linalg.LinAlgError, RuntimeWarning,):
            self.reset_covar()
            poseSample = np.random.multivariate_normal( self.pose, self.covar ) 
        # support = self.object_supporting_pose( poseSample )
        return Object( 
            label, 
            poseSample,
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
        if self.fresh:
            self.fresh = False
            return self.sample_symbol()
        else: 
            return self.sample_null()
    
    def sample_nothing( self, confuseProb = 0.1 ):
        """ Sample a negative indication for this pose """
        rtnObj = ObjectBelief()
        for i in range( len( _BLOCK_NAMES ) ):
            blkName_i = _BLOCK_NAMES[i]
            if blkName_i == _NULL_NAME:
                rtnObj.labels[ blkName_i ] = 1.0-confuseProb*(len( _BLOCK_NAMES )-1)
            else:
                rtnObj.labels[ blkName_i ] = confuseProb
        rtnObj.pose = np.array( self.pose )
        return rtnObj

    def update_pose_dist( self ):
        """ Update the pose distribution from the history of observations """
        self.fresh = True
        poseHist   = np.array( self.pHist )
        self.pHist = []
        nuPose     = np.mean( poseHist, axis = 0 )
        nuStdDev   = np.std( poseHist, axis = 0 )
        nuvar      = np.zeros( (_POSE_DIM,_POSE_DIM,) ) # ------ Pose covariance matrix
        for i, stdDev in enumerate( nuStdDev ):
            nuvar[i,i] = stdDev * stdDev
        self.pose = self.pose + np.dot(
            np.divide( 
                self.covar,
                np.add( self.covar, nuvar ), 
                where = self.covar != 0.0 
            ),
            np.subtract( nuPose, self.pose )
        )
        # print( self.covar )
        try:
            nuSum = np.add( 
                np.reciprocal( self.covar, where = self.covar != 0.0 ), 
                np.reciprocal( nuvar, where = nuvar != 0.0 ) 
            )
            self.covar = np.reciprocal( nuSum, where = nuSum != 0.0 )
        except RuntimeWarning:
            print( "WARNING: Covariance reset due to overflow!" )
            self.reset_covar()
    
    def integrate_belief( self, objBelief ):
        """ if `objBelief` is relevant, then Update this belief with evidence and return True, Otherwise return False """
        # NOTE: THIS WILL NOT BE AS CLEAN IF THE CLASSIFIER DOES NO PROVIDE A DIST ACROSS ALL CLASSES
        if self.p_pose_relevant( objBelief ):
            Nclass = len( _BLOCK_NAMES )
            cnfMtx = get_confusion_matx( Nclass )
            priorB = [ self.labels[ label ] for label in _BLOCK_NAMES ] 
            evidnc = [ objBelief.labels[ label ] for label in _BLOCK_NAMES ]
            updatB = multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc )
            self.labels = {}
            for i, name in enumerate( _BLOCK_NAMES ):
                self.labels[ name ] = updatB[i]
            self.pHist.append( objBelief.pose )
            if len( self.pHist ) >= _N_POSE_UPDT:
                self.update_pose_dist()
            return True
        else:
            return False