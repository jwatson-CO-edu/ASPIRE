########## DEV PLAN ################################################################################
"""
[Y] Combine `ObjectBelief` and `ObjectMemory` --> `ObjectMemory`, SIMPLIFY!, 2024-04-11: Testing req'd , 2024-04-12: Seems to work!
"""

########## INIT ####################################################################################

##### Imports #####

### Standard ###
import sys, pickle
from traceback import print_exc, format_exc
from pprint import pprint

### Special ###
import numpy as np
from py_trees.common import Status

### Local ###
from symbols import GraspObj, ObjPose
from utils import ( multiclass_Bayesian_belief_update, get_confusion_matx, get_confused_class_reading, 
                    DataLogger, origin_row_vec, )
from PB_BlocksWorld import PB_BlocksWorld
from actions import display_PDLS_plan
from env_config import ( _BLOCK_SCALE, _N_CLASSES, _CONFUSE_PROB, _NULL_NAME, _NULL_THRESH, 
                         _BLOCK_NAMES, _VERBOSE, _MIN_X_OFFSET, _ACTUAL_NAMES )

### PDDLStream ### 
sys.path.append( "../pddlstream/" )
from pddlstream.utils import read, INF, get_file_path
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.language.constants import print_solution, PDDLProblem
from pddlstream.algorithms.meta import solve


########## HELPER FUNCTIONS ########################################################################

def d_between_obj_poses( obj1, obj2 ):
    """ Calculate the translation between poses in the same frame """
    return np.linalg.norm( np.subtract( obj1.pose[:3], obj2.pose[:3] ) )


def sorted_obj_labels( obj ):
    """ Get the label dist keys in a PREDICTABLE ORDER """
    # WARNING: THIS FUNCTION BECOMES NECESSARY *AS SOON AS* GLOBAL LABLES ARE **NOT** FIXED!
    rtnLst = list( obj.labels.keys() )
    rtnLst.sort()
    return rtnLst


def extract_dct_values_in_order( dct, keyLst ):
    """ Get the `dct` values in the order specified in `keyLst` """
    rtnLst = []
    for k in keyLst:
        if k in dct:
            rtnLst.append( dct[k] )
    return rtnLst


def extract_class_dist_sorted_by_key( obj ):
    """ Get the discrete class distribution, sorted by key name """
    return np.array( extract_dct_values_in_order( obj.labels, sorted_obj_labels( obj ) ) )


def extract_class_dist_in_order( obj, order = _BLOCK_NAMES ):
    """ Get the discrete class distribution, in order according to environment variable """
    return np.array( extract_dct_values_in_order( obj.labels, order ) )


########## BELIEFS #################################################################################

class ObjectMemory:
    """ Attempt to maintain recent and constistent object beliefs based on readings from the vision system """

    def reset_beliefs( self ):
        """ Remove all references to the beliefs, then erase the beliefs """
        self.beliefs = []


    def __init__( self ):
        """ Set belief containers """
        self.reset_beliefs()

    
    def accum_evidence_for_belief( self, evidence, belief ):
        """ Use Bayesian multiclass update on `belief`, destructive """
        evdnc = extract_class_dist_in_order( evidence )
        prior = extract_class_dist_in_order( belief   )
        keys  = _BLOCK_NAMES
        pstrr = multiclass_Bayesian_belief_update( 
            get_confusion_matx( _N_CLASSES, confuseProb = _CONFUSE_PROB ), 
            prior, 
            evdnc 
        )
        for i, key in enumerate( keys ):
            belief.labels[ key ] = pstrr[i]


    def integrate_one_reading( self, objReading, maxRadius = 3.0*_BLOCK_SCALE ):
        """ Fuse this belief with the current beliefs """
        relevant = False

        # 1. Determine if this belief provides evidence for an existing belief
        dMin     = 1e6
        belBest  = None
        for belief in self.beliefs:
            d = d_between_obj_poses( objReading, belief )
            if (d < maxRadius) and (d < dMin):
                dMin     = d
                belBest  = belief
                relevant = True

        if relevant:
            belBest.visited = True
            self.accum_evidence_for_belief( objReading, belBest )
            # belBest.pose = np.array( objReading.pose ) # WARNING: ASSUME THE NEW NEAREST POSE IS CORRECT!
            belBest.pose = objReading.pose # WARNING: ASSUME THE NEW NEAREST POSE IS CORRECT!

        # 2. If this evidence does not support an existing belief, it is a new belief
        else:
            self.beliefs.append( objReading.copy() ) 

        # N. Return whether the reading was relevant to an existing belief
        return relevant
    

    def integrate_null( self, belief ):
        """ Accrue a non-observation """
        labels = get_confused_class_reading( _NULL_NAME, _CONFUSE_PROB, _BLOCK_NAMES )
        cnfMtx = get_confusion_matx( _N_CLASSES, _CONFUSE_PROB )
        priorB = [ belief.labels[ label ] for label in _BLOCK_NAMES ] 
        evidnc = [ labels[ label ] for label in _BLOCK_NAMES ]
        updatB = multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc )
        belief.labels = {}
        for i, name in enumerate( _BLOCK_NAMES ):
            belief.labels[ name ] = updatB[i]
    

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
            elif _VERBOSE:
                print( f"{str(belief)} DESTROYED!" )
        self.beliefs = retain


    def decay_beliefs( self ):
        """ Destroy beliefs that have accumulated too many negative indications """
        for belief in self.beliefs:
            if not belief.visited:
                self.integrate_null( belief )
        self.erase_dead()
        self.unvisit_beliefs()


    def belief_update( self, evdncLst ):
        """ Gather and aggregate evidence """

        ## Integrate Beliefs ##
        cNu = 0
        cIn = 0
        self.unvisit_beliefs()
        
        if not len( self.beliefs ):
            # WARNING: ASSUMING EACH OBJECT IS REPRESENTED BY EXACTLY 1 READING
            for objEv in evdncLst:
                self.beliefs.append( objEv.copy() )
        else:
            for objEv in evdncLst:
                if self.integrate_one_reading( objEv ):
                    cIn += 1
                else:
                    cNu += 1

        if _VERBOSE:
            if (cNu or cIn):
                print( f"\t{cNu} new object beliefs this iteration!" )
                print( f"\t{cIn} object beliefs updated!" )
            else:
                print( f"\tNO belief update!" )
        
        ## Decay Irrelevant Beliefs ##
        self.decay_beliefs()
        
        if _VERBOSE:
            print( f"Total Beliefs: {len(self.beliefs)}" )


    def most_likely_objects( self, N = 1 ):
        """ Get the `N` most likely combinations of object classes """

        def p_unique_labels( objLst ):
            """ Return true if there are as many classes as there are objects """
            lbls = set([sym.label for sym in objLst])
            if _NULL_NAME in lbls: # WARNING: HACK
                return False
            return len( lbls ) == len( objLst )

        ## Init ##
        comboList = [ [1.0,[],], ]
        ## Generate all class combinations with joint probabilities ##
        for bel in self.beliefs:
            nuCombos = []
            for combo_i in comboList:
                for label_j, prob_j in bel.labels.items():
                    prob_ij = combo_i[0] * prob_j

                    # objc_ij = GraspObj( label = label_j, pose = np.array( bel.pose ) )
                    objc_ij = GraspObj( label = label_j, pose = bel.pose )
                    
                    nuCombos.append( [prob_ij, combo_i[1]+[objc_ij,],] )
            comboList = nuCombos
        ## Sort all class combinations with decreasing probabilities ##
        comboList.sort( key = (lambda x: x[0]), reverse = True )
        ## Return top combos ##
        if N == 1:
            for combo in comboList:
                if p_unique_labels( combo[1] ):
                    return combo[1]
        elif N > 1:
            rtnCombos = []
            for i in range(N):
                rtnCombos.append( comboList[i][1] )
            return rtnCombos
        else:
            return list()



########## BASELINE PLANNER ########################################################################
_trgtRed = ObjPose( [ _MIN_X_OFFSET+2.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )
_trgtGrn = ObjPose( [ _MIN_X_OFFSET+6.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )


class BaselineTAMP:
    """ Basic TAMP loop against which the Method is compared """

    ##### Init ############################################################

    def reset_beliefs( self ):
        """ Erase belief memory """
        self.memory = ObjectMemory() # Distributions over objects


    def reset_state( self ):
        """ Erase problem state """
        self.status  = Status.INVALID
        self.symbols = []
        self.task    = None
        self.goal    = tuple()
        self.facts   = list()


    def __init__( self, world = None ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.reset_beliefs()
        self.world  = world if (world is not None) else PB_BlocksWorld()
        self.logger = DataLogger()
        # DEATH MONITOR
        self.noSoln =  0
        self.nonLim = 10


    def perceive_scene( self ):
        """ Integrate one noisy scan into the current beliefs """
        self.memory.belief_update( self.world.full_scan_noisy() )


    ##### Stream Creators #################################################

    def get_above_pose_stream( self ):
        """ Return a function that returns poses """

        def stream_func( *args ):
            """ A function that returns poses """

            if _VERBOSE:
                print( f"\nEvaluate ABOVE LABEL stream with args: {args}\n" )

            objcName = args[0]

            for sym in self.symbols:
                if sym.label == objcName:
                    # upPose = np.array( sym.pose )
                    upPose = sym.pose.copy()
                    # upPose[2] = 2.0*_BLOCK_SCALE
                    upPose.pose[2] = 2.0*_BLOCK_SCALE
                    print( f"FOUND a pose {upPose} supported by {objcName}!" )
                    yield (upPose,)

        return stream_func


    ##### Task And Motion Planning ########################################

    def pddlstream_from_problem( self ):
        """ Set up a PDDLStream problem with the UR5 """

        domain_pddl  = read( get_file_path( __file__, 'domain.pddl' ) )
        stream_pddl  = read( get_file_path( __file__, 'stream.pddl' ) )
        constant_map = {}
        stream_map = {
            ### Symbol Streams ###
            'sample-above': from_gen_fn( self.get_above_pose_stream() ), 
            # ### Symbol Tests ###
            # 2024-04-12: See if we can solve without this!
            # 'test-free-placment': from_test( self.get_free_placement_test() ),
        }

        if _VERBOSE:
            print( "About to create problem ... " )

        return PDDLProblem( domain_pddl, constant_map, stream_pddl, stream_map, self.facts, self.goal )


    def set_table( self ):
        """ Get ready for an experiment """
        self.world.robot.goto_home()
        self.world.reset_blocks()
        self.world.spin_for( 500 )


    def set_goal( self ):
        """ Set the goal """

        self.goal = ( 'and',
            
            ('GraspObj' , 'redBlock', _trgtRed  ), # ; Tower A
            ('Supported', 'ylwBlock', 'redBlock'), 
            ('Supported', 'bluBlock', 'ylwBlock'),

            ('GraspObj', 'grnBlock' , _trgtGrn  ), # ; Tower B
            ('Supported', 'ornBlock', 'grnBlock'), 
            ('Supported', 'vioBlock', 'ornBlock'),

            ('HandEmpty',),
        )

        if _VERBOSE:
            print( f"\n### Goal ###" )
            pprint( self.goal )
            print()


    def phase_1_Perceive( self, Nscans = 1 ):
        """ Take in evidence and form beliefs """

        for _ in range( Nscans ):
            planner.perceive_scene() # We need at least an initial set of beliefs in order to plan

        self.symbols = self.memory.most_likely_objects( N = 1 )
        self.status  = Status.RUNNING

        if _VERBOSE:
            print( f"\nStarting Objects:" )
            for obj in self.symbols:
                print( f"\t{obj}" )


    def phase_2_Conditions( self ):
        """ Get the necessary initial state, Check for goals already met """
        
        start = ObjPose( origin_row_vec() )

        self.facts = [
            ## Init Predicates ##
            ('Waypoint', start,),
            ('AtPose'  , start,),
            # ('Free'    , start,), # Not needed?
            ('HandEmpty',),
            ## Goal Predicates ##
            ('Waypoint', _trgtRed,),
            ('Free'    , _trgtRed,),
            ('Waypoint', _trgtGrn,),
            ('Free'    , _trgtGrn,),
        ] 

        for sym in self.symbols:
             self.facts.append( ('Graspable', sym.label,) )
             self.facts.append( ('Supported', sym.label, 'table',) )
             self.facts.append( ('GraspObj', sym.label, sym.pose,) )
             self.facts.append( ('Waypoint', sym.pose,) )

        if _VERBOSE:
            print( f"\n### Initial Symbols ###" )
            for sym in self.facts:
                print( f"\t{sym}" )
            print()

            
    def phase_3_Plan_Task( self ):
        """ Attempt to solve the symbolic problem """

        self.task = self.pddlstream_from_problem()

        self.logger.log_event( "Begin Solver" )

        # print( dir( self.task ) )
        if 0:
            print( f"\nself.task.init\n" )
            pprint( self.task.init )
            print( f"\nself.task.goal\n" )
            pprint( self.task.goal )
            print( f"\nself.task.domain_pddl\n" )
            pprint( self.task.domain_pddl )
            print( f"\nself.task.stream_pddl\n" )
            pprint( self.task.stream_pddl )

        try:
            
            solution = solve( 
                self.task, 
                algorithm      = "adaptive", #"focused", #"binding", #"incremental", #"adaptive", 
                unit_costs     = True, # False, #True, 
                unit_efforts   = True, # False, #True,
                reorder        = False,
                # initial_complexity = 4,
                # max_complexity = 4,
                # max_failures  = 4,
                # search_sample_ratio = 1/500, #1/1500, #1/5, #1/1000, #1/750 # 1/1000, #1/2000 #500, #1/2, # 1/500, #1/200, #1/10, #2, # 25 #1/25

            )

            print( "Solver has completed!\n\n\n" )
            print_solution( solution )
            
        except Exception as ex:
            self.logger.log_event( "SOLVER FAULT", format_exc() )
            self.status = Status.FAILURE
            print_exc()
            solution = (None, None, None)
            self.noSoln += 1 # DEATH MONITOR

        plan, cost, evaluations = solution
        if (plan is not None) and len( plan ):
            display_PDLS_plan( plan )
            self.currPlan = plan
            self.action   = get_BT_plan_until_block_change( plan, self.world )
            self.noSoln = 0 # DEATH MONITOR
        else:
            self.noSoln += 1 # DEATH MONITOR
            self.logger.log_event( "NO SOLUTION" )
            self.status = Status.FAILURE
        


########## MAIN ####################################################################################
if __name__ == "__main__":

    planner = BaselineTAMP()
    
    planner.set_goal()
    planner.logger.begin_trial()

    planner.phase_1_Perceive()
    planner.phase_2_Conditions()
    planner.phase_3_Plan_Task()

    planner.logger.end_trial( True )

    if 0:
        with open( "statistics/py3/magpie-tamp.pkl", 'rb' ) as f:
            postRunDiag = pickle.load( f )
            print( '\n' )
            pprint( postRunDiag )
            print( '\n' )

    if 0:
        for _ in range(10):
            planner.perceive_scene()
        for bel in planner.memory.beliefs:
            print( bel )

    if 0:
        objs = planner.world.full_scan_noisy()
        for obj in objs:
            print( obj )

    

    