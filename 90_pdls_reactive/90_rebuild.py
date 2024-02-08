########## DEV PLAN ################################################################################
"""
##### Planning #####
[Y] Rewrite `Object`, 2024-02-05: Seems correct!
[>] Rewrite Functions and Stream specs
    [Y] Object poses, 2024-02-05: Seems correct!
        [Y] Stream spec, 2024-02-05: Seems correct!
    [Y] Grasp effector pose from object pose, 2024-02-05: Seems correct!
        [Y] Stream spec, 2024-02-05: Unsure if the certification will create the correct object
    [Y] IK Soln from effector pose, 2024-02-06: Seems correct!
        [Y] Stream spec, 2024-02-06: Seems correct!
    [Y] FreePlacement, Checked by world, 2024-02-06: Seems correct!
        [Y] Stream spec, 2024-02-06: Seems correct!
    [Y] SafeTransit, Checked by world, 2024-02-06: Seems correct!
        [Y] Stream spec, 2024-02-06: Seems correct!
    [Y] SafeMotion, Checked by world, 2024-02-07: Seems correct!
        [Y] Stream spec, 2024-02-07: Seems correct!
[Y] Instantiate a PDLS world, 2024-02-07: No errors
[ ] Successful Planning
    [ ] Q: Can I ask the solver to be VERBOSE?

##### Execution #####
[ ] Rewrite Action drafts
    [ ] Place
    [ ] Move_Holding
    [ ] Pick
    [ ] Move_Free
[ ] Successful Plan Execution
    [ ] Test reactivity
[ ] Non-Reactive Version
"""



########## INIT ####################################################################################

##### Imports #####

### Standard ###
import sys, time, datetime, pickle, math
now = time.time
from random import random
from traceback import print_exc
# from collections import Counter
# from itertools import count

### Special ###
import numpy as np

import py_trees
from py_trees.common import Status
from py_trees.composites import Sequence

### Local ###

## PDDLStream ##
sys.path.append( "../pddlstream/" )
from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.language.generator import from_gen_fn, from_fn, empty_gen, from_test, universe_test
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object, negate_test
from pddlstream.language.constants import print_solution, PDDLProblem

## MAGPIE ##
sys.path.append( "../" )
# from magpie.poses import translation_diff

from utils import ( row_vec_to_pb_posn_ornt, pb_posn_ornt_to_row_vec, diff_norm, closest_dist_Q_to_segment_AB, )

from env_config import ( _GRASP_VERT_OFFSET, _GRASP_ORNT_XYZW, _NULL_NAME, _ACTUAL_NAMES, _MIN_X_OFFSET,
                         _NULL_THRESH, _BLOCK_SCALE, _CLOSEST_TO_BASE )
from pb_BT import connect_BT_to_robot_world
from PB_BlocksWorld import PB_BlocksWorld
from symbols import Pose, Config



########## BT-PLANNER INTERFACE ####################################################################

class BT_Runner:
    """ Run a BT with checks """

    def __init__( self, root, world, tickHz = 4.0 ):
        """ Set root node and world reference """
        self.root   = root
        self.world  = world
        self.status = Status.INVALID
        self.freq   = tickHz
        self.Nstep  = int( max(1.0, math.ceil((1.0 / tickHz) / world.period)))

    def setup_BT_for_running( self ):
        """ Connect the plan to world and robot """
        connect_BT_to_robot_world( self.root, self.world.robot, self.world )
        self.root.setup_with_descendants()

    def display_BT( self ):
        """ Draw the BT along with the status of all the nodes """
        py_trees.display.unicode_tree( root = self.root, show_status = True )

    def tick_once( self ):
        """ Run one simulation step """
        self.world.spin_for( self.Nstep )
        if self.status not in ( Status.FAILURE, Status.SUCCESS ):
            self.root.tick_once()
        self.status = self.root.status
        if self.status in ( Status.FAILURE, Status.SUCCESS ):
            self.display_BT()    

    


########## EXPERIMENT STATISTICS ###################################################################

class DataLogger:
    """ Keep track of when experiments begin and end """

    def __init__( self ):
        """ Setup stats dict """
        self.g_BGN   = None
        self.g_RUN   = False
        self.metrics = {
            "N"     : 0,
            "pass"  : 0,
            "fail"  : 0,
            "trials": [],
        }

    def begin_trial( self ):
        """ Increment number of trials and set state """
        self.g_BGN = now()
        self.g_RUN = True
        self.metrics['N'] += 1
        self.events = []

    def log_event( self, event ):
        """ Log a timestamped event """
        self.events.append( (now()-self.g_BGN, event,) )

    def end_trial( self, p_pass, infoDict = None ):
        """ Record makespan and trial info """
        if infoDict is None:
            infoDict = {}
        runDct = {
            "makespan" : now() - self.g_BGN,
            "result"   : p_pass,
            "events"   : self.events[:],
        }
        runDct.update( infoDict )
        self.metrics['trials'].append( runDct )
        if p_pass:
            self.metrics['pass'] += 1
        else:
            self.metrics['fail'] += 1

    def save( self, prefix = "Experiment-Data" ):
        """ Serialize recorded stats """
        fName = str( prefix ) + "__" + str( datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S') )
        with open( fName, 'wb' ) as handle:
            pickle.dump( self.metrics, handle )
        print( f"Wrote: {fName}" ) 

    def load( self, path ):
        """ De-serialize recorded stats """
        with open( path, 'rb' ) as handle:
            self.metrics = pickle.load( handle )
        return self.metrics


########## ACTIONS #################################################################################

    
# class GroundedAction( Sequence ):
#     """ This is the parent class for all actions available to the planner """

#     def __init__( self, objName, goal, world = None, robot = None, name = "Grounded Sequence" ):
#         super().__init__( name = name )
#         self.objName = objName # Type of object required
#         self.goal    = goal # -- Destination pose
#         self.symbol  = None # -- Symbol on which this behavior relies
#         self.msg     = "" # ---- Message: Reason this action failed -or- OTHER
#         self.ctrl    = robot # - Agent that executes
#         self.world   = world  #- Simulation ref

#     def get_grounded( self, symbol ):
#         """ Copy action with a symbol attached """
#         rtnAct = self.__class__( self.objName, self.goal, self.world, self.ctrl, self.name )
#         rtnAct.symbol = symbol
#         symbol.action = rtnAct
#         return rtnAct
    
#     def copy( self ):
#         """ Deep copy """
#         rtnObj = self.__class__( self.objName, self.goal, self.world, self.ctrl, self.name )
#         rtnObj.status = self.status
#         rtnObj.symbol = self.symbol
#         return rtnObj
    
#     def p_grounded( self ):
#         """ Return true if a symbol was assigned to this action """
#         return (self.symbol is not None)
    
#     def set_ground( self, symbol ):
#         """ Attach symbol """
#         self.symbol   = symbol
#         symbol.action = self

#     def cost( self ):
#         raise NotImplementedError( f"{self.name} REQUIRES a `cost` implementation!" )

#     def prep( self ):
#         raise NotImplementedError( f"{self.name} REQUIRES a `prep` implementation!" )
    
#     def validate_in_world( self ):
#         """ Check if the goal is already met """
#         return self.world.check_predicate( self.goal, _ACCEPT_POSN_ERR )


# class Pick( GroundedAction ):
#     """ BT that produces <OBJ@HAND> """

#     def __init__( self, objName, goal = None, world = None, robot = None, name = None ):
#         if name is None:
#             name = f"Pick: {objName} --> HAND"
#         super().__init__( objName, "HAND", world, robot, name )
    
#     def __repr__( self ):
#         """ String representation of the action """
#         return f"Pick: {self.objName} --> HAND"

#     def cost( self ):
#         """ Get the linear distance between the symbol pose and the destination """
#         robtPosn, robtOrnt = self.ctrl.get_current_pose()
#         return translation_diff( pb_posn_ornt_to_homog( robtPosn, robtOrnt ), row_vec_to_homog( self.symbol.pose ) )

#     def prep( self ):
#         """ Use the symbol grounding to parameterize the BT """

#         # 0. Check if the goal has already been met, PDLS FIXME
#         if self.validate_in_world():
#             print( f"{self.name} ALREADY DONE" )
#             self.status = Status.SUCCESS
#             return

#         # 1. Fetch ref to the object nearest the pose, if any
#         graspPose = self.symbol.pose
#         handle    = self.world.get_handle_at_pose( graspPose )

#         if handle is not None:
#             goalNam = self.world.get_handle_name( handle )

#             posnTgt, orntTgt = row_vec_to_pb_posn_ornt( graspPose )
            
#             posnTgt[2] += _GRASP_VERT_OFFSET
#             orntTgt = _GRASP_ORNT_XYZW.copy()

#             self.add_child( Pick_at_Pose( posnTgt, orntTgt, goalNam, zSAFE = _Z_SAFE, name = f"Pick_at_Pose: {graspPose}", 
#                                           ctrl = self.ctrl, world = self.world ) )
#         else:
#             self.status = Status.FAILURE
#             self.msg    = "Object miss"

        
# class Place( GroundedAction ):
#     """ BT that produces <OBJ@HAND> """

#     def __init__( self, objName, goal, world = None, robot = None, name = None ):
#         if name is None:
#             name = f"Place: {objName} --> {goal}"
#         super().__init__( objName, goal, world, robot, name )
    
#     def __repr__( self ):
#         """ String representation of the action """
#         return f"Place: {self.objName} --> {self.goal}"

#     def cost( self ):
#         """ Get the linear distance between the symbol pose and the destination """
#         robtPosn, robtOrnt = self.ctrl.get_current_pose()
#         return translation_diff( pb_posn_ornt_to_homog( robtPosn, robtOrnt ), row_vec_to_homog( self.goal ) )
    
#     def prep( self ):
#         """ Use the symbol grounding to parameterize the BT """

#         # 0. Check if the goal has already been met, PDLS FIXME
#         if self.validate_in_world():
#             print( f"{self.name} ALREADY DONE" )
#             self.status = Status.SUCCESS
#             return

#         # 1. Fetch ref to the object nearest the pose, if any
#         posnEnd, orntEnd = row_vec_to_pb_posn_ornt( self.goal )

#         if self.symbol is not None:
            
#             posnEnd[2] += _GRASP_VERT_OFFSET
#             orntEnd = _GRASP_ORNT_XYZW.copy()
#             self.add_child( Place_at_Pose( posnEnd, orntEnd, zSAFE = _Z_SAFE, name = "Place_at_Pose", 
#                                            ctrl = self.ctrl, world = self.world ) )
#         else:
#             self.status = Status.FAILURE
#             self.msg    = "Object miss"


# class Stack( GroundedAction ):
#     """ BT that produces <OBJ@HAND> """

#     def __init__( self, objName, goal, world = None, robot = None, name = None ):
#         if name is None:
#             name = f"Place: {objName} --> {goal}"
#         super().__init__( objName, goal, world, robot, name )
    
#     def __repr__( self ):
#         """ String representation of the action """
#         return f"Stack: {self.objName} --> {self.goal}"

#     def cost( self ):
#         """ Get the linear distance between the symbol pose and the destination """
#         robtPosn, robtOrnt = self.ctrl.get_current_pose()
#         return translation_diff( pb_posn_ornt_to_homog( robtPosn, robtOrnt ), row_vec_to_homog( self.goal ) )
    
#     def prep( self ):
#         """ Use the symbol grounding to parameterize the BT """

#         # 0. Check if the goal has already been met, PDLS FIXME
#         if self.validate_in_world():
#             print( f"{self.name} ALREADY DONE" )
#             self.status = Status.SUCCESS
#             return

#         # 1. Fetch ref to the object nearest the pose, if any
#         posnEnd, orntEnd = row_vec_to_pb_posn_ornt( self.goal )

#         if self.symbol is not None:
            
#             posnEnd[2] += _GRASP_VERT_OFFSET
#             orntEnd = _GRASP_ORNT_XYZW.copy()
#             self.add_child( Place_at_Pose( posnEnd, orntEnd, zSAFE = _Z_SAFE, name = "Place_at_Pose", 
#                                            ctrl = self.ctrl, world = self.world ) )
#         else:
#             self.status = Status.FAILURE
#             self.msg    = "Object miss"

########## PLANNER HELPERS #########################################################################

def prep_plan( plan ):
    """ Set appropriate targets """
    for action in plan:
        action.prep()

def p_plan_grounded( plan ):
    """ Return true if every action in the plan is grounded, Otherwise return False """
    for action in plan:
        if not action.p_grounded():
            return False
    return True

def plan_confidence( plan ):
    """ Return the least object label belief """
    belMin = 1e6
    for action in plan:
        prob   = action.symbol.prob()
        belMin = min( belMin, prob )
    return belMin

def plan_cost( plan ):
    """ Return the total cost of all actions """
    total = 0.0
    for action in plan:
        total += action.cost()
    return total

def release_plan_symbols( plan ):
    """ Detach symbols from all actions in the plan """
    for action in plan:
        if action.symbol is not None:
            action.symbol.action = None
            action.symbol = None



########## PLANS ###################################################################################


class Plan( Sequence ):
    """ Special BT `Sequence` with assigned priority, cost, and confidence """

    def __init__( self ):
        """ Set default priority """
        super().__init__()
        self.rank   = 0.0 # -------------- Priority of this plan
        self.rand   = random() * 10000.0 # Tie-breaker for sorting
        self.goal   = -1 # --------------- Goal that this plan satisfies if completed
        self.idx    = -1 # --------------- Current index of the running action
        self.msg    = "" # --------------- Message: Reason this plan failed -or- OTHER
        self.ctrl   = None
        self.world  = None

    def __getitem__( self, idx ):
        """ Return the child at `idx` """
        return self.children[ idx ]
    
    def __len__( self ):
        """ Return the number of children """
        return len( self.children )

    def append( self, action ):
        """ Add an action """
        self.add_child( action )

    def __lt__( self, other ):
        """ Compare to another plan """
        # Original Author: Jiew Meng, https://stackoverflow.com/a/9345618
        selfPriority  = (self.rank , self.rand )
        otherPriority = (other.rank, other.rand)
        return selfPriority < otherPriority
    
    def __repr__( self ):
        """ String representation of the plan """
        return f"<MockPlan, Goal: {self.goal}, Status: {self.status}, Index: {self.idx}>"

    def get_goal_spec( self ):
        """ Get a fully specified goal for this plan """
        raise NotImplementedError( "get_goal_spec" )
        # rtnGoal = []
        # for action in self:
        #     rtnGoal.append( Pose( None, action.objName, action.goal, _SUPPORT_NAME ) )
        # return rtnGoal
    


########## EXECUTIVE (THE METHOD) ##################################################################

class ReactiveExecutive:
    """ Least structure needed to compare plans """

    ##### Init ############################################################

    def reset_beliefs( self ):
        """ Erase belief memory """
        self.beliefs = [] # Distributions over objects
        self.symbols = []
        self.plans   = [] # PriorityQueue()

    def __init__( self, world ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.world = world
        self.reset_beliefs()

    ##### Belief Updates ##################################################

    def integrate_one_segmentation( self, objBelief ):
        """ Fuse this belief with the current beliefs """
        # 1. Determine if this belief provides evidence for an existing belief
        relevant = False
        for belief in self.beliefs:
            if belief.integrate_belief( objBelief ):
                relevant = True
                belief.visited = True
                # Assume that this is the only relevant match, break
                break
        # 2. If this evidence does not support an existing belief, it is a new belief
        if not relevant:
            self.beliefs.append( objBelief )
        return relevant

    def unvisit_beliefs( self ):
        """ Set visited flag to False for all beliefs """
        for belief in self.beliefs:
            belief.visited = False

    def decay_beliefs( self ):
        retain = []
        for belief in self.beliefs:
            if not belief.visited:
                belief.integrate_belief( belief.sample_nothing() )
            if belief.labels[ _NULL_NAME ] < _NULL_THRESH:
                retain.append( belief )
            else:
                print( f"{str(belief)} DESTROYED!" )

    def belief_update( self ):
        """ Gather and aggregate evidence """
        objEvidence = self.world.full_scan_noisy()
        print( f"Got {len(objEvidence)} beliefs!" )

        ## Integrate Beliefs ##
        cNu = 0
        cIn = 0
        self.unvisit_beliefs()
        for objEv in objEvidence:
            if self.integrate_one_segmentation( objEv ):
                cIn += 1
            else:
                cNu += 1
        if (cNu or cIn):
            print( f"\t{cNu} new object beliefs this iteration!" )
            print( f"\t{cIn} object beliefs updated!" )
        else:
            print( f"\tNO belief update!" )
        print( f"Total Beliefs: {len(self.beliefs)}" )

    def check_OOB( self, thresh_m = 10.0 ):
        """ Return true if any of the simulated objects are out of bounds """
        truSym = self.world.full_scan_true()
        for truth in truSym:
            posn, _ = row_vec_to_pb_posn_ornt( truth.pose )
            for coord in posn:
                if abs( coord ) >= thresh_m:
                    return True
        return False
    
    ##### Stream Creators #################################################
    # I hate this problem formulation so very much, We need a geometric grammar

    def get_object_stream( self ):
        """ Return a function that returns poses """

        def stream_func( *args ):
            """ A function that returns poses """

            print( f"\nEvaluate OBJECT stream with args: {args}\n" )

            objName = args[0]

            ## Sample Symbols ##
            self.belief_update()
            nuSym = [bel.sample_symbol() for bel in self.beliefs]
            # nuSym = [bel.sample_fresh() for bel in self.beliefs]
            foundSome = False
            for sym in nuSym:
                if sym.label != _NULL_NAME:
                    foundSome = True
                    print( f"OBJECT stream SUCCESS: {nuSym}\n" )
                    yield ( Pose( sym.pose ), ) 
                # if objName == sym.label:
                #     print( f"OBJECT stream SUCCESS\n" )
                #     yield ( Pose( sym.pose ), ) 
                # else yield nothing if we cannot certify the object!
            if not foundSome:
                print( f"OBJECT stream FAILURE\n" )
        return stream_func
    

    def get_grasp_stream( self ):
        """ Return a function that returns grasps """
        
        def stream_func( *args ):
            """ A function that returns grasps """
            
            print( f"\nEvaluate GRASP stream with args: {args}\n" )

            # targetPose = args[0].value
            label, pose = args
            grasp_pose = pose.value
            grasp_pose[2] += _GRASP_VERT_OFFSET
            posn, _ = row_vec_to_pb_posn_ornt( grasp_pose )
            ornt = _GRASP_ORNT_XYZW.copy()
            grasp_pose = Pose( pb_posn_ornt_to_row_vec( posn, ornt ) )
            # yield (Grasp(  Pose( targetPose ), Pose( grasp_pose ) ),)
            print( f"GRASP stream SUCCESS: {grasp_pose.value}\n" )
            yield (grasp_pose,) # FIXME: CHECK THAT THIS GETS CERTIFIED
             # else yield nothing if we cannot certify the object!

        return stream_func
    
    def get_IK_solver( self ):
        """ Return a function that computes Inverse Kinematics for a pose """

        def stream_func( *args ):
            """ A function that computes Inverse Kinematics for a pose """

            print( f"\nEvaluate IK stream with args: {args}\n" )

            effPose = args[0].value

            currPosn, _ = self.world.robot.get_current_pose()
            grspPosn, grspOrnt = row_vec_to_pb_posn_ornt( effPose )
            if diff_norm( currPosn, grspPosn ) < 2.0:
                graspQ = self.world.robot.calculate_ik_quat( grspPosn, grspOrnt )
                print( f"IK stream SUCCESS: {graspQ}\n" )
                yield ( Config( graspQ ), )
            else:
                print( f"IK stream FAILURE\n" )

        return stream_func
    
    def get_free_placement_test( self ):
        """ Return a function that checks if the pose is free from obstruction """

        def test_func( *args ):
            
            print( f"\nEvaluate PLACEMENT test with args: {args}\n" )

            label, pose = args
            posn , _    = row_vec_to_pb_posn_ornt( pose.value )

            ## Sample Symbols ##
            self.belief_update()
            nuSym = [bel.sample_symbol() for bel in self.beliefs]
            # nuSym = [bel.sample_fresh() for bel in self.beliefs]
            print( f"Symbols: {nuSym}" )
            for sym in nuSym:
                if label != sym.label:
                    symPosn, _ = row_vec_to_pb_posn_ornt( sym.pose )
                    if diff_norm( posn, symPosn ) < (2.0*_BLOCK_SCALE):
                        print( f"PLACEMENT test FAILURE\n" )
                        return False
            print( f"PLACEMENT test SUCCESS\n" )
            return True
        
        return test_func


    def get_safe_transit_test( self ):
        """ Return a function that checks if the path is free from obstruction """

        def test_func( *args ):
            
            print( f"\nEvaluate TRANSIT test with args: {args}\n" )

            # (bgn, end) = args
            label, bgn, end = args
            if diff_norm( bgn.value, end.value ) > 0.0: # FIXME, START HERE: USE THE VECTOR WAY!

                posnBgn, _ = row_vec_to_pb_posn_ornt( bgn.value )
                posnEnd, _ = row_vec_to_pb_posn_ornt( end.value )

                ## Sample Symbols ##
                self.belief_update()
                nuSym = [bel.sample_symbol() for bel in self.beliefs]
                # nuSym = [bel.sample_fresh() for bel in self.beliefs]
                print( f"Symbols: {nuSym}" )

                for sym in nuSym:
                    if sym.label != label:
                        Q, _ = row_vec_to_pb_posn_ornt( sym.pose )
                        d = closest_dist_Q_to_segment_AB( Q, posnBgn, posnEnd, True )    
                        if d < 2.0*_BLOCK_SCALE:
                            print( f"TRANSIT test FAILURE: {posnBgn}, {posnEnd}\n" )
                            return False
                print( f"TRANSIT test SUCCESS\n" )
                return True
            else:
                print( f"TRANSIT test SUCCESS\n" )
                return True
            
        return test_func
            

    def get_safe_motion_test( self ):
        """ Return a function that checks if the path is free from obstruction """

        def test_func( *args ):
            
            print( f"\nEvaluate MOTION test with args: {args}\n" )

            config1, config2 = args
            posn1  , _       = self.world.robot.fk_posn_ornt( config1.value )
            posn2  , _       = self.world.robot.fk_posn_ornt( config2.value )
            d = closest_dist_Q_to_segment_AB( [0.0,0.0,0.0], posn1, posn2, False )
            if math.isnan( d ):
                print( f"MOTION test SUCCESS: Non-intersection\n" )
                return True
            if diff_norm( posn1, posn2 ) <= 0.0:
                print( f"MOTION test SUCCESS\n" )
                return True
            if d < _CLOSEST_TO_BASE:
                print( f"MOTION test FAILURE: {d}\n" )
                return False
            else:
                print( f"MOTION test SUCCESS\n" )
                return True
            
        return test_func

    

    ##### PDLS Solver #####################################################

    def pddlstream_from_problem( self ):
        """ Set up a PDDLStream problem with the UR5 """

        domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
        constant_map = {}
        print( "Read files!" )

        print( 'Robot:', self.world.robot.get_name() )
        conf = Config( self.world.robot.get_joint_angles() )
        pose = Pose( pb_posn_ornt_to_row_vec( *self.world.robot.get_current_pose() ) )
        
        trgt = Pose( [ _MIN_X_OFFSET+2.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )
        # trgt = Pose( [ 0.492, 0.134, 0.600, 0.707, 0.0, 0.707, 0.0 ] )
        
        # tCnf = Config( [0 for _ in range(6)] )
        
        init = [
            ## Init Predicates ##
            ('Conf', conf),
            ('AtConf', conf),
            ('EffPose', pose),
            ('AtPose', pose),
            ('HandEmpty',),
            ## Goal Predicates ##
            ('Pose', trgt)
        ] # WARNING: UNBOUND POSE?
        
        print( f"### Initial Symbols ###" )
        for sym in init:
            print( f"\t{sym}" )

        print( "Robot grounded!" )

        for body in _ACTUAL_NAMES:
            init.append( ('Graspable', body) )

        goal = ('and',
                ('WObject', 'redBlock', trgt),
                ('HandEmpty',)
        )
        # goal = ('Holding', 'redBlock')  
        # goal = ('AtConf', tCnf)  
        # goal = ('AtPose', trgt)  

        """ # SAVE THIS FOR THE FINAL DEMO
        goal = ( 'and',
            ('AtPose', 'redBlock', Pose(None, 'redBlock', [ _MIN_X_OFFSET+2.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ])),
            ('AtPose', 'ylwBlock', Pose(None, 'ylwBlock', [ _MIN_X_OFFSET+4.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ])),
            ('Arched', 'bluBlock', 'redBlock', 'ylwBlock'),
        ) 
        """

        stream_map = {
            ### Symbol Streams ###
            'sample-object':      from_gen_fn( self.get_object_stream() ), 
            'sample-grasp':       from_gen_fn( self.get_grasp_stream()  ),
            'inverse-kinematics': from_gen_fn( self.get_IK_solver()     ),
            ### Symbol Tests ###
            'test-free-placment': from_test( self.get_free_placement_test() ),
            'test-safe-transit' : from_test( self.get_safe_transit_test()   ),
            'test-safe-motion':   from_test( self.get_safe_motion_test()    ),
        }

        print( "About to create problem ... " )
    
        return PDDLProblem( domain_pddl, constant_map, stream_pddl, stream_map, init, goal )

########## MAIN ####################################################################################

##### Env. Settings #####
np.set_printoptions( precision = 3, linewidth = 145 )


##### Run Sim #####
if __name__ == "__main__":

    ### Init ###

    world = PB_BlocksWorld()
    world.reset_blocks()
    robot = world.robot
    robot.goto_home()
    world.spin_for( 500 )

    planner = ReactiveExecutive( world )

    print( '\n\n\n##### PDLS INIT #####' )
    problem = planner.pddlstream_from_problem()
    print( 'Created!\n\n\n' )

    print( '##### PDLS SOLVE #####' )
    try:
        solution = solve( problem, 
                          algorithm = "incremental", #"focused", #"binding", #"incremental", #"adaptive", 
                          unit_costs = True, success_cost = 1,
                          visualize = True,
                          initial_complexity=4  )
        print( "Solver has completed!\n\n\n" )
    except Exception as ex:
        print( "SOLVER FAULT\n" )
        print_exc()
        solution = (None, None, None)
        print( "\n\n\n" )

    


    print_solution( solution )
    plan, cost, evaluations = solution

#     def exec_plans_noisy( self, N = 1200,  Npause = 200 ):
#         """ Execute partially observable plans """

#         self.world.reset_blocks()
#         self.world.spin_for( Npause )

#          # Number of iterations for this test
#         K =    5 # Number of top plans to maintain
#         ### Main Planner Loop ###  
#         currPlan     = None
#         achieved     = []
#         trialMetrics = Counter()
#         pPass        = False
#         pBork        = False
#         begin_trial()
#         # 2023-12-11: For now, loop a limited number of times
#         for i in range(N):

#             
            
#             ## Retain only fresh beliefs ##
#             belObj = []
#             for belief in self.beliefs:
#                 if belief.visited:
#                     belObj.append( belief )
#                 else:
#                     belief.integrate_belief( belief.sample_nothing() )
#                     
#             self.beliefs = belObj

#             

#             ## Ground Plans ##
#             svSym     = [] # Only retain symbols that were assigned to plans!
#             skeletons = [self.get_skeleton( j ) for j in range( len( self.skltns ) )]
#             for sym in nuSym:
#                 assigned = False
#                 for l, skel in enumerate( skeletons ):
#                     for j, action in enumerate( skel ):
#                         if not p_grounded( action ):
#                             if (action.objName == sym.label) and (not sym.p_attached()):
#                                 set_action_ground( action, sym )
#                                 assigned = True
#                         if assigned:
#                             break
#                     if assigned:
#                         break
#                 if sym.p_attached():
#                     svSym.append( sym )
                
#             for k, skel in enumerate( skeletons ):
#                 if p_plan_grounded( skel ):
#                     self.plans.append( skel )
#             self.symbols.extend( svSym )
#             print( f"There are {len(self.plans)} plans!" )

#             ## Grade Plans ##
#             savPln = []
#             for m, plan in enumerate( self.plans ):
#                 cost  = plan_cost( plan )
#                 prob  = plan_confidence( plan )
#                 score = cost - _LOG_PROB_FACTOR * log( prob, _LOG_BASE )
#                 plan.rank = score
#                 # Destroy (Degraded Plans || Plans with NaN Priority) #
#                 if (prob > _PLAN_THRESH) and (not isnan( score )):
#                     savPln.append( plan )
#                 else:
#                     release_plan_symbols( plan )
#                     print( f"\tReleased {len(plan)} symbols!" )

#             ## Enqueue Plans ##    
#             savPln.sort()
#             self.plans = savPln[:K]
#             for badPlan in savPln[K:]:
#                 release_plan_symbols( badPlan )
#             for m, plan in enumerate( self.plans ):
#                 print( f"\tPlan {m+1} --> Cost: {cost}, P = {prob}, {'Retain' if (prob > _PLAN_THRESH) else 'DELETE'}, Priority = {plan.rank}" )

#             ## Destroy Unlikely Symbols ##
#             savSym = [] # Only save likely symbols attached to plans
#             cDel   = 0
#             for sym in self.symbols:
#                 if (sym.prob() > _PLAN_THRESH) and sym.p_attached():
#                     savSym.append( sym )
#                 else:
#                     cDel += 1
#             self.symbols = savSym
#             print( f"Retained {len(self.symbols)} symbols, and deleted {cDel}!" )

#             ## Execute Current Plan ##
#             # Pop top plan
#             if (currPlan is None) and len( self.plans ):
#                 try:

#                     currPlan = self.plans[0]
#                     prep_plan( currPlan, self.world, self.world.robot )
#                     setup_plan_for_running( currPlan, self.world, self.world.robot )
#                     self.plans.pop(0)

#                     while currPlan.goal in achieved:
#                         if currPlan is not None:
#                             release_plan_symbols( currPlan )

#                         currPlan = self.plans[0]
#                         prep_plan( currPlan, self.world, self.world.robot )
#                         setup_plan_for_running( currPlan, self.world, self.world.robot )
#                         self.plans.pop(0)

#                 except (IndexError, AttributeError):
#                     if currPlan is not None:
#                         release_plan_symbols( currPlan )
#                     currPlan = None
#             if currPlan is not None:
#                 if currPlan.status == Status.SUCCESS:
#                     achieved.append( currPlan.goal )
#                     release_plan_symbols( currPlan )
#                     currPlan = None

#                 elif currPlan.status == Status.FAILURE:
#                     print( f"TRASHING failed plan: {currPlan}" )
#                     trialMetrics[ currPlan.msg ] += 1
#                     release_plan_symbols( currPlan )
#                     world.robot_release_all()
#                     currPlan = None

#                 elif plan_confidence( currPlan ) >= _PLAN_THRESH:
#                     # currPlan.tick( self.world, _ACCEPT_POSN_ERR )
#                     # print( currPlan.ctrl, currPlan.world )
                    
#                     ## Step ##
#                     self.world.spin_for( 10 )
#                     currPlan.tick_once()
#                     # currPlan.tick()

#                     if random() < _PROB_TICK_FAIL:
#                         currPlan.status = Status.FAILURE
#                         currPlan.msg    = "Action Fault"

#                 else:
#                     print( f"TRASHING unlikely plan: {currPlan}" )
#                     trialMetrics[ "Unlikely Symbol" ] += 1
#                     release_plan_symbols( currPlan )
#                     currPlan = None

#             ## Check Win Condition ##
#             nuChieved = []
#             for goalNum in achieved:
#                 skeleton = self.skltns[ goalNum ]
#                 goal     = skeleton.get_goal_spec()
#                 solved   = self.world.validate_goal_spec( goal, _ACCEPT_POSN_ERR )
#                 if solved:
#                     print( f"Goal {goalNum} is SOLVED!" )
#                     nuChieved.append( goalNum )
#                 else:
#                     trialMetrics[ "Goal NOT Met" ] += 1
#             achieved = nuChieved

#             if len( achieved ) >= len( self.skltns ):
#                 break

            
#             print()
            
#         pPass = (len( achieved ) >= len( self.skltns ))

#         if pPass:
#             print( "\n### GOALS MET ###\n" )
#         elif pBork:
#             print( "\n!!! SIM BORKED !!!\n" )
#         else:
#             print( "\n### TIMEOUT ###\n" )

#         for k, v in trialMetrics.items():
#             print( f"Failure: {k}, Occurrences: {v}" )

#         end_trial( pPass, trialMetrics )
        



#     planner = MockPlanner( world )
#     Nruns   = 250
    
#     ### Trials ###
#     for i in range( Nruns ):
#         print(f'\n##### Trial {i+1} of {Nruns} #####')
#         planner.exec_plans_noisy( 1200 )
#         world.spin_for( 200 )
#         print('\n')

#     ### Analyze ###
#     import matplotlib.pyplot as plt

#     print( f"Success Rate __ : {metrics['pass']/metrics['N']}" )
#     spans = [ dct['makespan'] for dct in metrics["trials"] ]
#     avgMs = sum( spans ) / metrics['N']
#     print( f"Average Makespan: {avgMs}" )

#     Nbins = 10

#     msPass = []
#     msFail = []

#     for trial in metrics["trials"]:
#         if trial['result']:
#             msPass.append( trial['makespan'] )
#         else:
#             msFail.append( trial['makespan'] )

#     with open( 'robotDemo250_2024-01-31.pkl', 'wb' ) as handle:
#         pickle.dump( metrics, handle )

#     with open( 'robotDemo250_2024-01-31_msPass.pkl', 'wb' ) as handle:
#         pickle.dump( msPass, handle )       

#     with open( 'robotDemo250_2024-01-31_msFail.pkl', 'wb' ) as handle:
#         pickle.dump( msFail, handle )    

#     plt.hist( [msPass, msFail], Nbins, histtype='bar', label=["Success", "Failure"] )

#     plt.legend(); plt.xlabel('Episode Makespan'); plt.ylabel('Count')
#     plt.savefig( 'robotDemo_Makespan.pdf' )

#     plt.show()
