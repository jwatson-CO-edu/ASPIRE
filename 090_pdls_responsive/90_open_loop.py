########## DEV PLAN ################################################################################
"""

##### Execution #####
[Y] Rewrite Action drafts
    [Y] Inspect plan output, 2024-02-08: Seems to make sense, but need to parse goals!
    [Y] Move_Free, 2024-02-08: Easy!
    [Y] Pick, 2024-02-08: Easy!
    [Y] Move_Holding, 2024-02-08: Easy!
    [Y] Place, 2024-02-08: Easy!
[Y] DOMAIN REFACTOR 5, Condense {Obj,Grasp,IKSoln} --into-> Waypoint, 2024-02-15: 30s for planning.
    [Y] Rewrite "domain.pddl", 2024-02-14: Seems consistent!
    [Y] Rewrite "stream.pddl", 2024-02-14: Seems consistent!
    [Y] Rewrite "sybmols.py", 2024-02-14: Condensed!
    [Y] Rewrite streams, 2024-02-14: Condensed!
    [Y] Test on 3 blocks, 2024-02-15: 1.5s-7s for planning!
    [Y] Test on 6 blocks, 2024-02-15: 30s for planning. Not bad, Not great. Commensurate with original paper results.
        [N] Does this problem need a setdown sprinkler?, 2024-02-15: Does not appear to be needed, but keep the option open
[>] Non-Reactive Version: Open Loop
    [Y] Check that the goal predicates are met, 2024-02-15: Straightforward
        [Y] (Obj ?label ?pose), 2024-02-13: Easy!
        [Y] (Holding ?label) ; From Pick, 2024-02-13: Easy!
        [Y] (HandEmpty) ; From Place, 2024-02-13: Easy!
        {P} ALL predicates?, 2024-02-13: Not at this time!
    [Y] Start with deterministic classes and poses, 2024-02-15: Solver req's about 30s for the 2-arch problem in the 6 block env
    [Y] Try two arches, 2024-02-15: Solver req's about 30s for the 2-arch problem in the 6 block env
    [Y] Re-implement sampling without dupes, 2024-02-18: Consistent belief query!
    [Y] Add a pre-check before the solver runs to verify that all the req'd objects EXIST, 2024-02-18: Consistent belief query!
    [>] Collect failure statistics
        [>] Generate graphs
         * No Sankey Graph for the open loop version
[ ] Replanning Version
    [ ] Working Demo @ PyBullet
        [ ] Plan open loop
        [ ] Execute one action
        [ ] Replan
        [ ] Loop until done or timeout
    [ ] Experimental Data
        [Y] What do I need to create a Sankey Graph? Is there a prettier one than PLT?, 2024-02-19: Will has the link
        [ ] Sankey Graph of sequence incidents during each episode
            [ ] Makespan on X-axis?
[ ] Responsive Version
    [ ] Working Demo @ PyBullet
        [ ] Goal check at the start of each symbolic phase --> Instantiate predicates
        [ ] Test 1: Replan after every action
        [ ] Test 2, if Test 1 results are poor: Replan ONLY after action FAILURES or crises of BELIEF
    [ ] Experimental Data
[ ] Data Analysis
[ ] PROPOSAL
"""



########## INIT ####################################################################################

##### Imports #####

### Standard ###
import sys, time, datetime, pickle, math
now = time.time
from random import random
from traceback import print_exc
from pprint import pprint
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
from pddlstream.language.function import FunctionInfo

## MAGPIE ##
sys.path.append( "../" )
# from magpie.poses import translation_diff

from utils import ( row_vec_to_pb_posn_ornt, pb_posn_ornt_to_row_vec, diff_norm, closest_dist_Q_to_segment_AB, )

from env_config import ( _GRASP_VERT_OFFSET, _GRASP_ORNT_XYZW, _NULL_NAME, _ACTUAL_NAMES, _MIN_X_OFFSET,
                         _NULL_THRESH, _BLOCK_SCALE, _CLOSEST_TO_BASE, _ACCEPT_POSN_ERR, _MIN_SEP, _Z_SAFE,
                         _N_POSE_UPDT, _WP_NAME, _SAMPLE_DET )
from pb_BT import connect_BT_to_robot_world, Move_Arm, Grasp, Ungrasp, pass_msg_up
from PB_BlocksWorld import PB_BlocksWorld, rand_table_pose
from symbols import Object, Path

from beliefs import ObjectMemory


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
        self.msg    = ""

    def setup_BT_for_running( self ):
        """ Connect the plan to world and robot """
        connect_BT_to_robot_world( self.root, self.world.robot, self.world )
        self.root.setup_with_descendants()

    def display_BT( self ):
        """ Draw the BT along with the status of all the nodes """
        print( py_trees.display.unicode_tree( root = self.root, show_status = True ) )

    def p_ended( self ):
        """ Has the BT ended? """
        return self.status in ( Status.FAILURE, Status.SUCCESS )

    def tick_once( self ):
        """ Run one simulation step """
        self.world.spin_for( self.Nstep )
        if not self.p_ended():
            self.root.tick_once()
        self.status = self.root.status
        if self.p_ended():
            pass_msg_up( self.root )
            self.msg = self.root.msg
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
            "events"   : list( self.events ),
        }
        self.events = []
        runDct.update( infoDict )
        self.metrics['trials'].append( runDct )
        if p_pass:
            self.metrics['pass'] += 1
        else:
            self.metrics['fail'] += 1

    def save( self, prefix = "Experiment-Data" ):
        """ Serialize recorded stats """
        fName = str( prefix ) + "__" + str( datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S') ) + ".pkl"
        with open( fName, 'wb' ) as handle:
            pickle.dump( self.metrics, handle )
        print( f"Wrote: {fName}" ) 

    def load( self, path ):
        """ De-serialize recorded stats """
        with open( path, 'rb' ) as handle:
            self.metrics = pickle.load( handle )
        return self.metrics


########## ACTIONS #################################################################################

class GroundedAction( Sequence ):
    """ This is the parent class for all actions available to the planner """

    def __init__( self, args = None, goal = None, world = None, robot = None, name = "Grounded Sequence" ):
        super().__init__( name = name )
        self.args   = args if (args is not None) else list() # Prerequisites required by this action
        self.goal   = goal if (goal is not None) else list() # Predicates satisfied by this action
        self.symbol = None # -- Symbol on which this behavior relies
        self.msg    = "" # ---- Message: Reason this action failed -or- OTHER
        self.ctrl   = robot # - Agent that executes
        self.world  = world  #- Simulation ref

    def get_grounded( self, symbol ):
        """ Copy action with a symbol attached """
        rtnAct = self.__class__( self.goal, self.world, self.ctrl, self.name )
        rtnAct.symbol = symbol
        symbol.action = rtnAct
        return rtnAct
    
    def copy( self ):
        """ Deep copy """
        rtnObj = self.__class__( self.goal, self.world, self.ctrl, self.name )
        rtnObj.status = self.status
        rtnObj.symbol = self.symbol
        return rtnObj
    
    def p_grounded( self ):
        """ Return true if a symbol was assigned to this action """
        return (self.symbol is not None)
    
    def set_ground( self, symbol ):
        """ Attach symbol """
        self.symbol   = symbol
        symbol.action = self

    def cost( self ):
        raise NotImplementedError( f"{self.name} REQUIRES a `cost` implementation!" )

    def prep( self ):
        raise NotImplementedError( f"{self.name} REQUIRES a `prep` implementation!" )
    
    def __repr__( self ):
        """ Get the name, Assume child classes made it sufficiently descriptive """
        return str( self.name )



class MoveFree( GroundedAction ):
    """ Move the unburdened effector to the given location """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        # ?obj1 ?obj2 ?traj
        obj1, obj2, traj = args

        if name is None:
            name = f"Move Free to {obj2.pose}"
        super().__init__( args, goal, world, robot, name )
    
        try:
            for x_i in traj.wp[1:]:
                grasp_pose = list( x_i.grasp )
                posn, ornt = row_vec_to_pb_posn_ornt( grasp_pose )
                self.add_child( 
                    Move_Arm( posn, ornt, name = name, ctrl = robot, world = world )
                )
        except Exception as e:
            print( f"MoveFree ERROR, Args: {args}",  )


class Pick( GroundedAction ):
    """ Add object to the gripper payload """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        # ?label ?obj
        label, obj = args
        
        if name is None:
            name = f"Pick object {label} at {obj.pose}"
        super().__init__( args, goal, world, robot, name )

        self.add_child( 
            # Grasp( label, name = name, ctrl = robot, world = world )
            Grasp( label, obj.pose, name = name, ctrl = robot, world = world )
        )


class MoveHolding( GroundedAction ):
    """ Move the burdened effector to the given location """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        # ?label ?objBgn ?objEnd ?traj
        label, objBgn, objEnd, traj = args

        if name is None:
            name = f"Move Holding {label} to {objEnd.pose}"
        super().__init__( args, goal, world, robot, name )

        # Move grasp to every waypoint in the trajectory
        for x_i in traj.wp[1:]:
            grasp_pose = list( x_i.grasp )
            posn, ornt = row_vec_to_pb_posn_ornt( grasp_pose )
            self.add_child( 
                Move_Arm( posn, ornt, name = name, ctrl = robot, world = world )
            )


class Place( GroundedAction ):
    """ Let go of gripper payload """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        # ?label ?obj
        label, obj = args
        
        if name is None:
            name = f"Place object {label} at {obj.pose}"
        super().__init__( args, goal, world, robot, name )

        self.add_child( 
            Ungrasp( name = name, ctrl = robot, world = world )
        )


class Stack( GroundedAction ):
    """ Let go of gripper payload """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        # ?labelUp ?labelDn1 ?labelDn2 ?objUp ?objDn1 ?objDn2
        labelUp, labelDn1, labelDn2, objUp, objDn1, objDn2 = args
        
        if name is None:
            name = f"Place object {labelUp} on top of {labelDn1} and {labelDn2} at {objUp.pose}"
        super().__init__( args, goal, world, robot, name )

        self.add_child( 
            Ungrasp( name = name, ctrl = robot, world = world )
        )

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


########## PDDL PARSING ############################################################################


########## PLANS ###################################################################################

class Plan( Sequence ):
    """ Special BT `Sequence` with assigned priority, cost, and confidence """

    def __init__( self ):
        """ Set default priority """
        super().__init__( name = "PDDLStream Plan" )
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

def get_BT_plan_from_PDLS_plan( pdlsPlan, world ):
    """ Translate the PDLS plan to one that can be executed by the robot """
    rtnBTlst = []
    if pdlsPlan is not None:
        for i, pdlsAction in enumerate( pdlsPlan ):
            actName  = pdlsAction.name
            actArgs  = pdlsAction.args
            btAction = None
            if actName == "move_free":
                btAction = MoveFree( actArgs, goal=None, world = world, robot=world.robot )
            elif actName == "pick":
                btAction = Pick( actArgs, goal=None, world = world, robot=world.robot )
            elif actName == "move_holding":
                btAction = MoveHolding( actArgs, goal=None, world = world, robot=world.robot )
            elif actName == "place":
                btAction = Place( actArgs, goal=None, world = world, robot=world.robot )
            elif actName == "stack":
                btAction = Stack( actArgs, goal=None, world = world, robot=world.robot )
            else:
                raise NotImplementedError( f"There is no BT procedure defined for a PDDL action named {actName}!" )
            print( f"Action {i+1}, {actName} --> {btAction.name}, planned!" )
            rtnBTlst.append( btAction )
    rtnPlan = Plan()
    rtnPlan.add_children( rtnBTlst )
    return rtnPlan
    
def display_PDLS_plan( plan ):
    print( f"\nPlan output from PDDLStream:" )
    if plan is not None:
        for i, action in enumerate( plan ):
            # print( dir( action ) )
            print( f"\t{i+1}: { action.__class__.__name__ }, {action.name}" )
            for j, arg in enumerate( action.args ):
                print( f"\t\tArg {j}:\t{type( arg )}, {arg}" )
    else:
        print( plan )

def p_list_duplicates( lst ):
    """ Return True if a value appears more than once """
    s = set( lst )
    return (len( lst ) > len( s ))

def move_cost_func( *args ):
    """ Return the distance between the endpoints [m] """
    
    print( f"\nEvaluate MOVEMENT cost with args: {args}\n" )

    traj = args[0]
    bgn  = traj.wp[ 0]
    end  = traj.wp[-1]
    dist = diff_norm( bgn.pose[:3], end.pose[:3] )

    print( f"\nMOVEMENT cost: {dist}\n" )

    return dist

########## EXECUTIVE (THE METHOD) ##################################################################

class ResponsiveExecutive:
    """ Least structure needed to compare plans """

    ##### Init ############################################################

    def reset_beliefs( self ):
        """ Erase belief memory """
        self.memory  = ObjectMemory() # Distributions over objects
        self.symbols = []
        self.plans   = [] # PriorityQueue()
        self.last    = {}

    def __init__( self, world = None ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.world = world if (world is not None) else PB_BlocksWorld()
        self.reset_beliefs()

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

    def sample_determ( self, objName ):
        """ Return the deterministic pose of the block """
        nuSym = self.world.full_scan_true()
        for sym in nuSym:
            if sym.label not in self.last:
                self.last[ sym.label ] = sym
        if objName in self.last:
            return self.last[ objName ]
        else:
            return None
        

    def calc_grasp( self, objPose ):
        """ A function that returns grasps """
        grasp_pose = list( objPose )
        grasp_pose[2] += _GRASP_VERT_OFFSET
        posn, _ = row_vec_to_pb_posn_ornt( grasp_pose )
        ornt = _GRASP_ORNT_XYZW.copy()
        return pb_posn_ornt_to_row_vec( posn, ornt )
    
        
    def calc_ik( self, effPose ):
        """ Helper function for IK and Path planners """
        currPosn, _        = self.world.robot.get_current_pose()
        grspPosn, grspOrnt = row_vec_to_pb_posn_ornt( effPose )
        if diff_norm( currPosn, grspPosn ) < 2.0:
            return self.world.robot.calculate_ik_quat( grspPosn, grspOrnt )
        else:
            return None
        

    def object_from_label_pose( self, objcName, objcPose ):
        """ Load a Waypoint with relevant data """
        grspPose = self.calc_grasp( objcPose )
        grspCnfg = self.calc_ik( grspPose )
        rtnObj   = Object( objcName, objcPose )
        rtnObj.grasp  = grspPose
        rtnObj.config = grspCnfg
        return rtnObj


    def add_grasp_config_to_object( self, obj ):
        """ Load an existing Waypoint with relevant data """
        obj.grasp  = self.calc_grasp( obj.pose )
        obj.config = self.calc_ik( obj.grasp )
    

    def get_object_stream( self ):
        """ Return a function that returns poses """

        def stream_func( *args ):
            """ A function that returns poses """

            print( f"\nEvaluate OBJECT stream with args: {args}\n" )

            objcName = args[0]

            ## Sample Symbols ##
            if _SAMPLE_DET:
                rtnObj = self.sample_determ( objcName )
            else:
                # rtnObj = self.memory.sample_consistent( objcName )
                rtnObj = self.memory.sample_consistent_fresh( objcName )
            if (rtnObj is not None) and (rtnObj.label != _NULL_NAME):
                self.add_grasp_config_to_object( rtnObj )
                print( f"OBJECT stream SUCCESS: {rtnObj}\n" )
                yield (rtnObj,)

        return stream_func
    

    def safe_motion_test( self, bgn, end ):
        """ Test if a line between two effector positions passes through the robot base """
        posn1  , _       = row_vec_to_pb_posn_ornt( bgn.grasp )
        posn2  , _       = row_vec_to_pb_posn_ornt( end.grasp )
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
    

    def path_segment_checker( self, bgn, end ):
        """ Helper function for the path planner stream """

        print( bgn, end )
        if not self.safe_motion_test( bgn, end ):
            return False

        label      = bgn.label
        posnBgn, _ = row_vec_to_pb_posn_ornt( bgn.pose )
        posnEnd, _ = row_vec_to_pb_posn_ornt( end.pose )

        if label == _WP_NAME:
            return True

        if diff_norm( posnBgn, posnEnd ) > 0.0: 

            ## Sample Symbols ##
            if _SAMPLE_DET:
                nuSym = self.world.full_scan_true()
            else:
                nuSym = self.memory.scan_consistent_fresh()
                # nuSym = self.memory.scan_consistent()
            print( f"Symbols: {nuSym}" )

            for sym in nuSym:
                if sym.label != label:
                    Q, _ = row_vec_to_pb_posn_ornt( sym.pose )
                    d = closest_dist_Q_to_segment_AB( Q, posnBgn, posnEnd, True )    
                    if d < _MIN_SEP:
                        return False
            return True
        else:
            return True


    def get_path_planner( self ):
        """ Return a function that checks if the path is free from obstruction """

        def stream_func( *args, fluents=[] ):
            
            print( f"\nEvaluate PATH stream with args: {args}\n" )
            print( f"\nEvaluate PATH stream with fluents: {fluents}\n" )

            obj1, obj2 = args

            print( obj1, obj2 )
            if self.path_segment_checker( obj1, obj2 ):
                yield ( Path( [obj1, obj2],), )
            else:
                posnBgn, orntEnd = row_vec_to_pb_posn_ornt( obj1.pose )
                posnEnd, _       = row_vec_to_pb_posn_ornt( obj2.pose )
                posnMid    = np.add( posnBgn, posnEnd ) / 2.0
                posnMid[2] = _Z_SAFE
                orntMid    = list( orntEnd )
                objcPose = pb_posn_ornt_to_row_vec( posnMid, orntMid )
                mid      = self.object_from_label_pose( obj1.label, objcPose )
                if self.path_segment_checker( obj1, mid ) and self.path_segment_checker( mid, obj2 ):
                    yield ( Path( [obj1, mid, obj2,],), )
        return stream_func
    

    def get_carry_planner( self ):
        """ Return a function that checks if the path is free from obstruction """

        def stream_func( *args, fluents=[] ):
            
            print( f"\nEvaluate CARRY stream with args: {args}\n" )
            print( f"\nEvaluate CARRY stream with fluents: {fluents}\n" )

            label, obj1, obj2 = args

            # if self.path_segment_checker( obj1, obj2 ):
            #     yield ( Path( [obj1, obj2],), )
            # else:
            posnBgn, orntEnd = row_vec_to_pb_posn_ornt( obj1.pose )
            posnEnd, _       = row_vec_to_pb_posn_ornt( obj2.pose )
            posnMid    = np.add( posnBgn, posnEnd ) / 2.0
            posnMid[2] = _Z_SAFE
            orntMid    = orntEnd[:]
            objcPose = pb_posn_ornt_to_row_vec( posnMid, orntMid )
            mid      = self.object_from_label_pose( obj1.label, objcPose )
            if self.path_segment_checker( obj1, mid ) and self.path_segment_checker( mid, obj2 ):
                yield ( Path( [obj1, mid, obj2,],), )
        return stream_func
    

    def get_stacker( self ):
        """ Return a function that computes Inverse Kinematics for a pose """

        def stream_func( *args, fluents=[] ):
            """ A function that computes a stacking pose across 2 poses """

            print( f"\nEvaluate STACK PLACE stream with args: {args}\n" )
            print( f"\nEvaluate STACK PLACE stream with fluents: {fluents}\n" )

            objDn1, objDn2 = args

            posnDn1, orntDn1 = row_vec_to_pb_posn_ornt( objDn1.pose )
            posnDn2, _       = row_vec_to_pb_posn_ornt( objDn2.pose )
            dist   = diff_norm( posnDn1, posnDn2 )
            sepMax = 2.1*_BLOCK_SCALE
            zMax   = max( posnDn1[2], posnDn2[2] )
            if (dist <= sepMax) and (zMax < 1.25*_BLOCK_SCALE) and (objDn1.label != objDn2.label):
                posnUp    = np.add( posnDn1, posnDn2 ) / 2.0
                posnUp[2] = 2.0*_BLOCK_SCALE
                orntUp    = orntDn1[:]
                objUp     = self.object_from_label_pose( _WP_NAME, pb_posn_ornt_to_row_vec( posnUp, orntUp ) )
                print( f"STACK PLACE stream SUCCESS: {objUp.pose}\n" )
                yield (objUp,)
            else:
                print( f"STACK PLACE stream FAILURE: {dist} > {sepMax}, {zMax} > {1.1*_BLOCK_SCALE}, {objDn1.label} == {objDn2.label} \n" )

        return stream_func
    

    def get_free_placement_test( self ):
        """ Return a function that checks if the pose is free from obstruction """

        def test_func( *args ):
            
            print( f"\nEvaluate PLACEMENT test with args: {args}\n" )

            # label, pose = args
            label, obj = args
            posn , _    = row_vec_to_pb_posn_ornt( obj.pose )

            ## Sample Symbols ##
            if _SAMPLE_DET:
                nuSym = self.world.full_scan_true()
            else:
                # nuSym = self.memory.scan_consistent()
                nuSym = self.memory.scan_consistent_fresh()
            print( f"Symbols: {nuSym}" )

            for sym in nuSym:
                if label != sym.label:
                    symPosn, _ = row_vec_to_pb_posn_ornt( sym.pose )
                    if diff_norm( posn, symPosn ) < ( _MIN_SEP ):
                        print( f"PLACEMENT test FAILURE\n" )
                        return False
            print( f"PLACEMENT test SUCCESS\n" )
            return True
        
        return test_func

    ##### Function Creators ###############################################

    def get_movement_cost_function( self ):
        """ Return a function that estimates the path length """

        def move_cost_func( *args ):
            """ Return the distance between the endpoints [m] """
            
            print( f"\nEvaluate MOVEMENT cost with args: {args}\n" )

            traj = args[0]
            bgn  = traj.wp[ 0]
            end  = traj.wp[-1]
            dist = diff_norm( bgn.pose[:3], end.pose[:3] )

            print( f"\nMOVEMENT cost: {dist}\n" )

            return dist
        
        return move_cost_func


    ##### Goal Validation #################################################

    def validate_predicate( self, pred ):
        """ Check if the predicate is true """
        pTyp = pred[0]
        if pTyp == 'HandEmpty':
            print( f"HandEmpty: {self.world.grasp}" )
            return (len( self.world.grasp ) == 0)
        elif pTyp == 'GraspObj':
            pLbl = pred[1]
            pPos = pred[2]
            tObj = self.world.get_block_true( pLbl )
            print( pred )
            print( "GraspObj:", pPos.pose[:3], tObj.pose[:3] )
            print( f"GraspObj: {diff_norm( pPos.pose[:3], tObj.pose[:3] )} <= {_ACCEPT_POSN_ERR}" )
            return (diff_norm( pPos.pose[:3], tObj.pose[:3] ) <= _ACCEPT_POSN_ERR)
        elif pTyp == 'Supported':
            lblUp = pred[1]
            lblDn = pred[2]
            objUp = self.world.get_block_true( lblUp )
            objDn = self.world.get_block_true( lblDn )
            xySep = diff_norm( objUp.pose[:2], objDn.pose[:2] )
            zSep  = objUp.pose[2] - objDn.pose[2] # Signed value
            print( f"Supported, X-Y Sep: {xySep} <= {2.0*_BLOCK_SCALE}, Z Sep: {zSep} >= {1.35*_BLOCK_SCALE}" )
            return ((xySep <= 2.0*_BLOCK_SCALE) and (zSep >= 1.35*_BLOCK_SCALE))
        else:
            print( f"UNSUPPORTED predicate check!: {pTyp}" )
            return False
    
    def validate_goal( self, goal ):
        """ Check if the goal is met """
        if goal[0] == 'and':
            for g in goal[1:]:
                if not self.validate_predicate( g ):
                    return False
            return True
        else:
            raise ValueError( f"Unexpected goal format!: {goal}" )

    ##### PDLS Solver #####################################################

    def check_goal_objects( self, goal, symbols ):
        """ Return True if the labels mentioned in the goals are a subset of the determinized symbols """
        goalSet = set([])
        symbSet = set( [sym.label for sym in symbols] )
        for g in goal:
            if isinstance( g, (tuple, list) ):
                prdName = g[0]
                if prdName == 'GraspObj':
                    goalSet.add( g[1] )
                elif prdName == 'Supported':
                    goalSet.add( g[1] )
                    goalSet.add( g[2] )
                else:
                    continue
        return (goalSet <= symbSet)

    def pddlstream_from_problem( self ):
        """ Set up a PDDLStream problem with the UR5 """

        domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
        # print( type( domain_pddl ) )

        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))

        # exit(0)

        constant_map = {}
        print( "Read files!" )

        print( 'Robot:', self.world.robot.get_name() )
        start = self.object_from_label_pose( _WP_NAME, pb_posn_ornt_to_row_vec( *self.world.robot.get_current_pose() ) )
        
        trgtRed = self.object_from_label_pose( 'redBlock', [ _MIN_X_OFFSET+2.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )

        trgtYlw = self.object_from_label_pose( 'ylwBlock', [ _MIN_X_OFFSET+4.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )

        trgtBlu = self.object_from_label_pose( 'bluBlock', [ _MIN_X_OFFSET+6.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )

        trgtGrn = self.object_from_label_pose( 'grnBlock', [ _MIN_X_OFFSET+6.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )

        trgtOrn = self.object_from_label_pose( 'ornBlock', [ _MIN_X_OFFSET+8.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )

        trgtVio = self.object_from_label_pose( 'vioBlock', [ _MIN_X_OFFSET+7.0*_BLOCK_SCALE, 0.000, 2.0*_BLOCK_SCALE,  1,0,0,0 ] )
        
        init = [
            ## Init Predicates ##
            ('Waypoint', start),
            ('AtObj', start),
            ('HandEmpty',),
            ## Goal Predicates ##
            ('Waypoint', trgtRed),
            ('Waypoint', trgtYlw),
            # ('Waypoint', trgtBlu),
            ('Waypoint', trgtGrn),
            ('Waypoint', trgtOrn),
        ] 
        
        for body in _ACTUAL_NAMES:
            init.append( ('Graspable', body) )

        print( f"### Initial Symbols ###" )
        for sym in init:
            print( f"\t{sym}" )

        print( "Robot grounded!" )

        # goal = ('and', ('AtObj', trgtRed ), )
        # goal = ('and', ('Holding', 'redBlock'),  )

        # goal = ('and',
        #         ('GraspObj', 'ylwBlock', trgtYlw),
        #         ('GraspObj', 'redBlock', trgtRed),
        #         ('GraspObj', 'bluBlock', trgtBlu),
        #         ('HandEmpty',)
        # )
        
        goal = ( 'and',
            ('HandEmpty',),
            
            ('GraspObj', 'redBlock', trgtRed),
            ('GraspObj', 'ylwBlock', trgtYlw),
            ('Supported', 'bluBlock', 'redBlock'),
            ('Supported', 'bluBlock', 'ylwBlock'),

            ('GraspObj', 'grnBlock', trgtGrn),
            ('GraspObj', 'ornBlock', trgtOrn),
            ('Supported', 'vioBlock', 'grnBlock'),
            ('Supported', 'vioBlock', 'ornBlock'),
        ) 

        stream_map = {
            ### Symbol Streams ###
            'sample-object':    from_gen_fn( self.get_object_stream() ), 
            'find-safe-motion': from_gen_fn( self.get_path_planner()  ),
            'find-safe-carry':  from_gen_fn( self.get_carry_planner() ),
            'find-stack-place': from_gen_fn( self.get_stacker()       ),
            ### Symbol Tests ###
            'test-free-placment': from_test( self.get_free_placement_test() ),
            ### Cost Functions ###
            # 'MoveCost': from_fn( self.get_movement_cost_function() )
            # 'MoveCost': move_cost_func
            # 'MoveCost': self.get_movement_cost_function()
        }

        print( "About to create problem ... " )
    
        return PDDLProblem( domain_pddl, constant_map, stream_pddl, stream_map, init, goal )
    

    def run_one_episode( self, plan, logger = None ):
        """ Run a single experiment and collect data """
        
        btPlan = get_BT_plan_from_PDLS_plan( plan, self.world )
        print( "\n\n\n" )

        btr = BT_Runner( btPlan, self.world, 20.0 )
        btr.setup_BT_for_running()

        while not btr.p_ended():
            btr.tick_once()
            if (btr.status == Status.FAILURE):
                logger.log_event( "Action Failure: " + btr.msg )

        return (btr.status == Status.SUCCESS)


    def run_N_episodes( self, N ):
        """ Run N experiments and collect statistics """
        
        logger = DataLogger()
        robot  = self.world.robot

        for i in range( N ):
            print( f"\n\n########## Experiment {i+1} of {N} ##########" )

            robot.goto_home()
            self.world.reset_blocks()
            self.world.spin_for( 500 )

            self.reset_beliefs()

            logger.begin_trial()

            print( '\n\n\n##### PDLS INIT #####' )
            problem = self.pddlstream_from_problem()

            if not _SAMPLE_DET:
                for _ in range( 3*_N_POSE_UPDT+1 ):
                    self.memory.belief_update( self.world.full_scan_noisy() ) # We need at least an initial set of beliefs in order to plan

                objs = self.memory.scan_consistent_fresh()
                # objs = self.memory.scan_consistent()
                print( f"Starting Objects:" )
                for obj in objs:
                    print( f"\t{obj}" )
                    

                if not self.check_goal_objects( problem.goal, objs ):
                    logger.log_event( "Required objects missing" )   
                    logger.end_trial( False, { 'symbols' : [str(sym) for sym in self.memory.scan_consistent_fresh()] } ) 
                    # logger.end_trial( False, { 'symbols' : [str(sym) for sym in self.memory.scan_consistent()] } ) 
                    continue


            print( f"\n##### Solving Problem {i+1} #####" )

            
            logger.log_event( "Begin Solver" )

            problem = self.pddlstream_from_problem()


            try:
                # stream_info = {
                #     'MoveCost': FunctionInfo(opt_fn=self.get_movement_cost_function()),
                # }
                # 'ff-eager-pref': 20s
                # 'ff-ehc' : No Sol'n
                # 'goal-lazy' : Very long
                # 'dijkstra' : Very long
                # 'max-astar' : Very long
                # 'lmcut-astar' : No Sol'n
                # 'ff-astar' : 40s
                # 'ff-ehc' : Very long
                # 'ff-wastar1' : 30s
                # 'ff-wastar2' : 15s
                # 'ff-wastar4' : 10-15s, Fails sometimes
                # 'ff-wastar5' : 10-15s, Fails sometimes
                # 'cea-wastar1' : Fails often
                # 'cea-wastar3' : 15-20s, Fails sometimes
                # 'cea-wastar5' : Very long
                # 'ff-wastar3' : 7-15s

                planner = 'ff-wastar3' #'ff-eager-pref' # 'add-random-lazy' # 'ff-eager-tiebreak' #'goal-lazy' #'ff-eager'
                solution = solve( 
                    problem, 
                    algorithm = "adaptive", #"focused", #"binding", #"incremental", #"adaptive", 
                    max_skeletons = 50,
                    max_time      = 80.0,
                    unit_costs   = False, 
                    unit_efforts = False,
                    effort_weight = 10.0, #200.0, #100.0, #50.0, #20.0, # 5.0, # 2.0 #10.0,
                    success_cost = 40,
                    initial_complexity = 1,
                    complexity_step = 1,
                    search_sample_ratio = 1/1000, #1/1500, #1/5, #1/1000, #1/750 # 1/1000, #1/2000 #500, #1/2, # 1/500, #1/200, #1/10, #2, # 25 #1/25
                    reorder = False, # Setting to false bare impacts sol'n time
                    planner = planner
                    # stream_info = stream_info,
                )
                print( "Solver has completed!\n\n\n" )
                print_solution( solution )
            except Exception as ex:
                print( "SOLVER FAULT\n" )
                print_exc()
                solution = (None, None, None)
                print( "\n\n\n" )

            logger.log_event( "End Solver" )

            plan, cost, evaluations = solution
            display_PDLS_plan( plan )
            
            print( "\n\n\n" )

            self.run_one_episode( plan, logger )

            # if not _SAMPLE_DET:
            #     for i in range( 2*_N_POSE_UPDT+1 ):
            #         self.belief_update() # We need at least an initial set of beliefs in order to plan

            goalMet = self.validate_goal( problem.goal )

            print( f"Were the goals met?: {goalMet}" )

            logger.end_trial( goalMet )

        logger.save( "Open-Loop_6-blocks_" )
        



########## MAIN ####################################################################################
import os

##### Env. Settings #####
np.set_printoptions( precision = 3, linewidth = 130 )


##### Run Sim #####
if __name__ == "__main__":
    planner = ResponsiveExecutive()
    planner.run_N_episodes( 5 )

    print("\n\nExperiments DONE!!\n\n")
    # duration =   3  # seconds
    # freq     = 500 #440  # Hz
    # os.system( 'play -nq -t alsa synth {} sine {}'.format(duration, freq) )
    
