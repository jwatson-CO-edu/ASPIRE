########## DEV PLAN ################################################################################
"""
PLOVER: [P]robabilistic [L]anguage [O]ver [V]olumes for [E]nvironment [R]easoning
In which I entertain an obsession with Geometric Grammars without knowledge of its Ultimate Value

[>] Solve the 2-Arches problem in the 6-Block environemnt
    [>] Belief
        [>] Belief Samples
    [ ] Required Symbols and Predicates (w/ PyBullet tests)
        [ ] Block class
        [ ] Block beliefs
        [ ] Object @ Location predicate
        [ ] Object On Object predicate
            [ ] Object Above Object predicate
            [ ] Object Touching Object predicate
            {?} Object Supported by Object(s) predicate, WARNING: NOT MVP
        [ ] Robot Holding Object predicate
    [ ] Action Components
        [ ] Lightweight Pre-Image Volume
            [ ] Start with convex hull, This is a simple problem with few obstructions
            {?} Move to Minkowski Sums only if needed
            [ ] Render preimage to PyBullet
        [ ] Define and Project a presumptive future state
            [ ] Q: How to automatically sample targets that fulfill the req'd predicates?  Can this come directly from the predicate definition?
            [ ] Q: How to differentiate a presumptive future fact from a present fact?
                [ ] Q: Is there a need to establish durations in order to prevent collisions during planning?
            [ ] Render presumptive state(s) to PyBullet
    [ ] Required Actions
        [ ] Move Arm
            [ ] Automatically resolve Move Free -vs- Move Holding w/ predicate
        [ ] Pick
        [ ] Place
            [ ] Automatically resolve Stacked On/Touching relationships
    [ ] Build Geometric Solver via increasingly complex problems, Execute Open Loop
        * Sample randomly. Do NOT optimize until it is ABSOLUTELY REQUIRED!
        [ ] Object @ Location
            [ ] Planned
            [ ] Execute Open Loop
        [ ] Object On Object
            [ ] Planned
                * Sample in regions that will ground the desired predicate
            [ ] Execute Open Loop
        [ ] 1 Arch
            [ ] Planned
                [ ] Q: What is a general and efficient way to satisfy two `On` predicates simultaneously? 
                    Where to sample? How to compute where to samples based on {Predicates, Goals}?
            [ ] Execute Open Loop
        [ ] 2 Arches
            [ ] Planned
                [ ] Q: Does it make sense to automatically break a problem into subproblems?
                [ ] Q: Is there an efficient means to determine if subproblems interfere with each other?
            [ ] Execute Open Loop
    [ ] Full TAMP Loop
        [ ] Collect data, esp. on solver performance
        [ ] Demonstrate superiority over PDLS
    [ ] Full MAGPIE Loop: MAGPIE and PLOVER are friends!

    {?} DANGER, NON-MVP: Intutive Interface
        [ ] Render Predicates in 3D
        [ ] Render Plan in 3D
        [ ] Render Faults in 3D
        [ ] JSON Goal Statement
    
[ ] Show Them, Show Them All (They Called Me Mad)
    [ ] Assess Graduation Risk
    [ ] Refine PLOVER slide deck
        * A representation with desirable properties
            - Scene graphs are maps that robots can plan on
            - Probabilistic scene graphs are maps that robots can plan on probabilistically
            - Desirable Properties, Clear and Correct representation of
                * Pose
                * Pose Uncert
                * Relationships
                * Relationship Uncert
        * Facts should be physical first, and semantic second, (Cats Don't Study Semantics)            
            - Symbols (concepts) can be fuzzy
            - Predicates (relationships) can be fuzzy
                * Partially met
                * Uncertain
        * How can a fact be physical? --> By expressing it geometrically!
            - When we can measure the degree to which a fact is true, We can optimize on degree 
              (Callback to degree of completion in Prelim)
            - When we can measure our confidence in the truth of a fact, We can optimize on confidence
        * PDDL -vs- PLOVER Showdown: What it is like to solve the same problem in both frameworks?
            - Compare symbols
            - Compare predicates
            - Compare solver performance
                * Solver: Running time, Success rate
                * Execution: Running time, Success rate
        * Explainability in Robot Plans
            - Geometric expression of facts has a side effect of being able to *render* facts to a display
            - If we can render facts, then we get visual explainability for (almost) free!
        * Render the Physical Facts, Intuitively
            - Show facts from 2-Arches, and how they change over time
            - Show facts from Fruit Picking, and how they change over time
        * Future Work
            - Allow the human to correct or create robot plans in a 3D env, Compare to MoveIt!
            - PLOVER planning in other metric spaces other than the physical (Word2Vec???)
                * Polytopes, distance, translation, and rotation (geo alg) all work the same in higher dims!
            - What is the connection to SLAM methods that operate on scene graphs?
            - What is the connection to VQA systems that operate on scene graphs?
                * Feed a visual scene to an LLM and let it explain the scene
            - Can PLOVER simplify working on hi-dim problems?
    [ ] Demo
        [ ] Performant Planning
        [ ] Intutive Output and Troubleshooting
        {?} Human intervention?
    [ ] Choose audience
    [ ] Market PLOVER: Share what is exciting and true in a concise way

[ ] Model the fruit picking problem: There are MANY questions to be answered!
    [ ] Q: Need to handle novel object classes?
        [ ] Q: Can an object class remain indeterminate until it is identified with certainty?
    [ ] Q: How to handle objects for which a model DNE?
        [ ] Q: Does it make sense to *build* a model?
    [ ] Q: Does this require shape completion?
    [ ] Q: What are the LLM connections?
        [ ] Q: Can geo predicates provide input for VQA?
        [ ] Q: Can the LLM suggest geo predicates?
    
[ ] Model the ConMod problem

[ ] DANGER: Review the PLOVER `DEV PLAN` at https://github.com/jwatson-CO-edu/CogArch/blob/main/README.md#plover-rich-real-world-representation
    [ ] Q: How to model unintended/unmodeled side effects?
        [ ] Q: How to even identify them?
"""
########## INIT ####################################################################################

##### Imports #####

### Standard ###
from uuid import uuid5
from random import random

### Special ###
import numpy as np
from trimesh import Trimesh
from scipy.stats import chi2

### Local ###
from env_config import _POSN_STDDEV, _ORNT_STDDEV
from utils import ( p_lst_has_nan, pb_posn_ornt_to_row_vec, row_vec_normd_ornt )



########## COMPONENTS ##############################################################################


class Volume:
    """ Basic geometric representation """

    def __init__( self ):
        """ Geo Data """
        self.mesh = Trimesh()



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
    """ A physical thing that the robot has beliefs about """

    ##### Init ############################################################

    def reset_pose_distrib( self ):
        """ Reset the pose distribution """
        self.stddev = [_POSN_STDDEV for _ in range(3)]
        self.stddev.extend( [_ORNT_STDDEV for _ in range(4)] )

    def __init__( self, label = "", pose = None, volume = None ):
        """ Set pose Gaussian and geo info """
        super().__init__( label, pose, volume )
        self.symbols = {}
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
        # FIXME, START HERE: NORMALIZE THE QUAT PART `row_vec_normd_ornt`
        return 
