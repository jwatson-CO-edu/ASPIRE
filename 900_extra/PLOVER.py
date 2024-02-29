"""
PLOVER.py
[P]robabilistic [L]anguage [O]ver [V]olumes for [E]nvironment [R]easoning: Representation + Solver + Planner
James Watson
Version 2024-03
Please see "DEV_PLAN.md" for information, design approach, and suggested applications.
This software comes with no guarantee or warranty of correctness or functionality; neither explicit nor implied!
"""
########## INIT ####################################################################################

##### Imports #####

### Standard ###
import time
from uuid import uuid4
from random import random
from math import isnan

### Special ###
import numpy as np
import trimesh
# from trimesh.viewer.windowed import SceneViewer
from scipy.stats import chi2

### Local ###
from env_config import ( _PRIOR_POS_S, _PRIOR_ORN_S, _NULL_NAME, _CONFUSE_PROB, _NULL_THRESH, 
                         _EXIST_THRESH, _MIN_SEP, )
from utils import ( p_lst_has_nan, roll_outcome, get_confusion_matx,
                    multiclass_Bayesian_belief_update )
from components import Volume
from geometry import sample_pose, diff_norm, pose_covar

##### Settings #####
np.set_printoptions( precision = 5 )

########## SCENE GRAPH #############################################################################


class SpatialNode:
    """ A concept that can be situated in space and participate in relationships """
    # NOTE: This can also be used as a non-physical reference frame

    def __init__( self, label = "", pose = None, volume = None ):
        self.ID       = uuid4() # --------------------------------------- Means for identifying an unique object
        self.label    = label # ----------------------------------------- Text label, possibly non-unique
        self.pose     = pose if (pose is not None) else [0,0,0,1,0,0,0] # Absolute pose
        self.relPose  = [0,0,0,1,0,0,0] # ------------------------------- Relative pose
        self.volume   = volume if (volume is not None) else Volume() # -- Defining volume in space
        self.data     = {} # -------------------------------------------- TBD
        self.incoming = {} # -------------------------------------------- Upstream 
        self.outgoing = {} # -------------------------------------------- Downstream
        self.t_create = time.time() # ----------------------------------- Time when this object instantiated
        self.dead     = False # ----------------------------------------- Item not relevant, Mark for cleaning
        # TDB: Give nodes a lifespan so that we avoid collisions with them when sequencing actions?
        # BEGIN TIME?
        # END TIME?

    def add_below( self, objOrListOrDict ):
        """ Add an object, or a collection of objects, as outgoing nodes, return number added """
        if isinstance( objOrListOrDict, list ):
            for obj in objOrListOrDict:
                self.outgoing[ obj.ID ] = obj
            return len( objOrListOrDict )
        if isinstance( objOrListOrDict, dict ):
            self.outgoing.update( objOrListOrDict )
            return len( objOrListOrDict )
        self.outgoing[ objOrListOrDict.ID ] = objOrListOrDict
        return 1

    # 2024-02-28: For now, allow Python to take care of the actual garbage collection of orphan node memory.
    #             Avoid GC terminology so as not to confuse users.
    
    def clean_downstream( self ):
        """ Erase all `dead` objects downstream of this node, recursively """
        delIDs = []
        for nID, node in self.outgoing.items():
            node.clean_downstream()
            if node.dead:
                delIDs.append( nID )
        for dID in delIDs:
            del self.outgoing[ dID ]
        


class Object( SpatialNode ):
    """ A determinized instance of an object belief """

    def __init__( self, label = "", pose = None, volume = None, ref = None ):
        """ Set pose Gaussian and geo info """
        super().__init__( label, pose, volume )
        self.ref = ref

    def p_ref( self ):
        """ Is this symbol connected to the belief that spawned it? """
        return (self.ref is not None)

    def prob( self ):
        """ Query the parent belief to calc the current likelihood of this symbol """
        if self.p_ref():
            # Calc the joint prob of pose and label, given the current dist of belief that spawned self
            try:
                pr = self.ref.prob_density( self ) * self.ref.labels[ self.label ]
            except Exception as e:
                print(e)
                return 0.0
            return pr
        else:
            return 0.0
        
    def __repr__( self ):
        """ Symbol <OBJ@LOC,P> """
        return f"<Object {self.label} @ {self.pose}, Pr={self.prob()}>"



########## BELIEF ##################################################################################


class ObjectBelief( SpatialNode ):
    """ The concept of a physical thing that the robot has beliefs about """

    ##### Init ############################################################

    def reset_pose_distrib( self ):
        """ Reset the pose distribution """
        self.stddev = [ _PRIOR_POS_S for _ in range(3)] # Standard deviation of pose
        self.stddev.extend( [_PRIOR_ORN_S for _ in range(4)] )


    def __init__( self, label = "", pose = None, volume = None ):
        """ Set pose Gaussian and geo info """
        super().__init__( label, pose, volume )
        self.symbols = {} # -- All symbols sampled from this object concept
        self.labels  = {} # -- Discrete dist over labels for this object
        self.pHist   = [] # -- Recent history of poses
        self.visited = False # Has there been evidence for this pose
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
        
    def prob_density( self, obj ):
        """ Return the probability that this object lies within the present distribution """
        x     = np.array( obj.pose )
        mu    = np.array( self.pose )
        sigma = pose_covar( self.stddev )
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
            # posnSample = np.random.multivariate_normal( self.pose, self.pose_covar() ) 
            poseSample = sample_pose( self.pose, self.stddev ) 
        except (np.linalg.LinAlgError, RuntimeWarning,):
            self.reset_pose_distrib()
            # posnSample = np.random.multivariate_normal( self.pose, self.pose_covar() ) 
            poseSample = np.random.multivariate_normal( self.pose, self.stddev ) 
        while p_lst_has_nan( poseSample ):
            self.reset_std_dev()
            poseSample = np.random.multivariate_normal( self.pose, self.stddev ) 
        return poseSample
    

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
        if self.p_reading_relevant( objReading ):
            allKeys = set( objReading.labels.keys() )
            allKeys.update( self.labels.keys() )
            keyList = list( allKeys )
            keyList.sort()
            priorL = [ self.labels[k] if (k in self.labels) else 0.0 for k in keyList ]
            evidnc = [ objReading.labels[ k ] if (k in objReading.labels) else 0.0 for k in keyList ]
            Nclass = len( keyList )
            cnfMtx = get_confusion_matx( Nclass, _CONFUSE_PROB )
            updatB = multiclass_Bayesian_belief_update( cnfMtx, priorL, evidnc )
            self.labels = {}
            for i, name in enumerate( keyList ):
                self.labels[ name ] = updatB[i]
            return True
        else:
            return False
    

    def integrate_null( self ):
        """ Accrue a non-observation """
        Nclass  = len( self.labels )
        labels  = {}
        
        if _NULL_NAME not in self.labels:
            self.labels[ _NULL_NAME ] = 0.0

        ordLbls = self.sorted_labels()

        for objName_i in ordLbls:
            if objName_i == _NULL_NAME:
                # WARN: `_CONFUSE_PROB` SHOULD NOT BE A CONSTANT THING AND COMES FROM THE SPECIFIC CLASSIFIER
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


    def belief_from_reading( self, objReading ):
        """ Center a new belief on the incoming reading """
        nuBelief = ObjectBelief()
        nuBelief.labels = dict( objReading.labels )
        nuBelief.pose   = np.array( objReading.pose )
        return nuBelief
    

    def integrate_one_reading( self, objReading ):
        """ Fuse this belief with the current beliefs """
        # 1. Determine if this belief provides evidence for an existing belief
        relevant = False
        for belief in self.beliefs:
            if belief.integrate_reading( objReading ):
                relevant = True
                belief.visited = True
                # Assume that this is the only relevant match, break
                break
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
                belief.dead = True
                print( f"{str(belief)} DESTROYED!" )
        self.beliefs = retain

    
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


    def scan_max_likelihood( self, Nsample = 3 ):
        """ Get a list of unique samples, Keep only the most likely version of a label """
        smplSym = []
        for _ in range( Nsample ):
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
    

    def p_symbols_credible( self, objs ):
        """ Return true only if our belief in every symbol is above the trash threshold """
        for obj in objs:
            if obj.prob() < _EXIST_THRESH:
                return False
        return True



    def scan_consistent( self ):
        """ Get the most likely samples that do not collide with one another """
        uniqSym = self.scan_max_likelihood()
        while (not self.p_noncolliding( uniqSym )) or (not self.p_symbols_credible( uniqSym )):
            uniqSym = self.scan_max_likelihood()
            print( "RESCAN!" ) # Warning that we are in an infinite loop
        return uniqSym
    


########## PROBABILISTIC LANGUAGE OVER VOLUMES for ENVIRONMENT REASONING ###########################
# Please see "DEV_PLAN.md" for information, design approach, and applications
    
class PLOVER:
    """ Representation + Solver + Planner """

    def __init__( self ):
        """ Instantiate memory and reasoning """
        self.memory  = ObjectMemory() # ---------- Hybrid symbols with pose and class uncert
        self.graph   = SpatialNode( "LabFrame" ) # Root node for global probabilistic scene graph


    def belief_update( self, objEvidence ):
        """ Wrapper for memory update """
        self.memory.belief_update( objEvidence )


    def sample_all( self, eraseOld = True ):
        """ Wrapper for a consistent scan (non-colliding, prob above thresh) of all objects in memory """
        self.samples = self.memory.scan_consistent()
        # FIXME: ERASE OLD BELIEFS AND SYMBOLS FROM THE SCENE GRAPH, ADD NEW AT LAB FRAME
        return list( self.samples )



########## GRAPHICS OUTPUT #########################################################################

class VolumeScene:
    """ Render PLOVER beliefs and facts to the screen """

    def __init__( self ):
        """ Set up graphics for the scene graph """
        self.scene = trimesh.Scene()

    def add_node( self, node ):
        """ Add the determinized/mean volume to the scene to be rendered """
        self.scene.add_geometry( node.volume )

    def show( self ):
        """ Create a `trimesh` window and render the scene """
        # https://gist.github.com/wkentaro/a2e92f6e52c418080c00ef8992c46e37
        # FIXME, START HERE: SHOW
