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
                         _NULL_THRESH, _BLOCK_SCALE, _CLOSEST_TO_BASE, _ACCEPT_POSN_ERR, _MIN_SEP, _Z_SAFE )
from pb_BT import connect_BT_to_robot_world, Move_Arm, Grasp, Ungrasp
from PB_BlocksWorld import PB_BlocksWorld, rand_table_pose
from symbols import Pose, Config, Path, Object



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
    
    def validate_in_world( self ):
        """ Check if the goal is already met """
        for prdct in self.goal:
            if not self.world.check_predicate( prdct, _ACCEPT_POSN_ERR ):
                return False
        return True
    
    def __repr__( self ):
        """ Get the name, Assume child classes made it sufficiently descriptive """
        return str( self.name )
    
# 2024-02-08: Ignore goal parsing & checking until all necessary actions execute in PB
# FIXME: ADD GOAL PARSING
# FIXME: ADD GOAL CHECKING


class MoveFree( GroundedAction ):
    """ Move the unburdened effector to the given location """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        # ?effPose1 ?effPose2 ?config1 ?config2
        effPose1, effPose2, config1, config2 = args

        if name is None:
            name = f"Move Free to {effPose2}"
        super().__init__( args, goal, world, robot, name )
    
        posn, ornt = row_vec_to_pb_posn_ornt( effPose2.value )

        self.add_child( 
            Move_Arm( posn, ornt, name = name, ctrl = robot, world = world )
        )


class Pick( GroundedAction ):
    """ Add object to the gripper payload """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        # ?label ?pose ?effPose ?config
        label, pose, effPose, config = args
        
        if name is None:
            name = f"Pick object {label} at {pose}"
        super().__init__( args, goal, world, robot, name )

        self.add_child( 
            Grasp( label, name = name, ctrl = robot, world = world )
        )


class MoveHolding( GroundedAction ):
    """ Move the burdened effector to the given location """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        # ?label ?poseBgn ?poseEnd ?effPose1 ?effPose2 ?config1 ?config2 ?traj
        label, bgnPose, endPose, effPose1, effPose2, config1, config2, traj = args

        if name is None:
            name = f"Move Holding {label} to {endPose}"
        super().__init__( args, goal, world, robot, name )

        # Compute a grasp for every waypoint in the trajectory
        for x_i in traj.x[1:]:
            grasp_pose = x_i.value
            grasp_pose[2] += _GRASP_VERT_OFFSET
            posn, _ = row_vec_to_pb_posn_ornt( grasp_pose )
            ornt = _GRASP_ORNT_XYZW.copy()

            self.add_child( 
                Move_Arm( posn, ornt, name = name, ctrl = robot, world = world )
            )


class Place( GroundedAction ):
    """ Let go of gripper payload """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        label, pose, effPose = args
        
        if name is None:
            name = f"Place object {label} at {pose}"
        super().__init__( args, goal, world, robot, name )

        self.add_child( 
            Ungrasp( name = name, ctrl = robot, world = world )
        )

class Stack( GroundedAction ):
    """ Let go of gripper payload """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        labelUp, labelDn1, labelDn2, poseDn1, poseDn2, poseUp, effPose = args
        
        if name is None:
            name = f"Place object {labelUp} on top of {labelDn1} and {labelDn2} at {poseUp}"
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
    


########## EXECUTIVE (THE METHOD) ##################################################################

class ReactiveExecutive:
    """ Least structure needed to compare plans """

    ##### Init ############################################################

    def reset_beliefs( self ):
        """ Erase belief memory """
        self.beliefs = [] # Distributions over objects
        self.symbols = []
        self.plans   = [] # PriorityQueue()
        self.last    = {}

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

    def sample_fresh( self, label ):
        """ Maintain a pose dict and only update it when needed """
        nuSym = [bel.sample_fresh() for bel in self.beliefs]
        for sym in nuSym:
            if (sym.label != _NULL_NAME):
                self.last[ sym.label ] = Pose( sym.pose )
        if label in self.last:
            return self.last[ label ]
        else:
            return None

    def get_object_stream( self ):
        """ Return a function that returns poses """

        def stream_func( *args ):
            """ A function that returns poses """

            print( f"\nEvaluate OBJECT stream with args: {args}\n" )

            objName = args[0]

            ## Sample Symbols ##
            rtnPose = self.sample_fresh( objName )
            if rtnPose is not None:
                print( f"OBJECT stream SUCCESS: {rtnPose.value}\n" )
                yield ( rtnPose, ) 
            else:
                print( f"OBJECT stream FAILURE: No {objName}\n" )

        return stream_func
    

    def get_grasp_stream( self ):
        """ Return a function that returns grasps """
        
        def stream_func( *args ):
            """ A function that returns grasps """
            
            print( f"\nEvaluate GRASP stream with args: {args}\n" )

            label, pose = args
            grasp_pose = pose.value
            grasp_pose[2] += _GRASP_VERT_OFFSET
            posn, _ = row_vec_to_pb_posn_ornt( grasp_pose )
            ornt = _GRASP_ORNT_XYZW.copy()
            grasp_pose = Pose( pb_posn_ornt_to_row_vec( posn, ornt ) )
            print( f"GRASP stream SUCCESS: {grasp_pose.value}\n" )
            yield (grasp_pose,) 

        return stream_func
    

    def calc_ik( self, effPose ):
        """ Helper function for IK and Path planners """
        currPosn, _        = self.world.robot.get_current_pose()
        grspPosn, grspOrnt = row_vec_to_pb_posn_ornt( effPose.value )
        if diff_norm( currPosn, grspPosn ) < 2.0:
            graspQ = self.world.robot.calculate_ik_quat( grspPosn, grspOrnt )
            return Config( graspQ )
        else:
            return None


    def get_IK_solver( self ):
        """ Return a function that computes Inverse Kinematics for a pose """

        def stream_func( *args ):
            """ A function that computes Inverse Kinematics for a pose """

            print( f"\nEvaluate IK stream with args: {args}\n" )

            effPose = args[0]
            graspQ  = self.calc_ik( effPose )
            if graspQ is not None:
                print( f"IK stream SUCCESS: {graspQ}\n" )
                yield (graspQ,)
            else:
                print( f"IK stream FAILURE: Excessive distance\n" )

        return stream_func
    
    
    def path_segment_checker( self, label, bgn, end ):
        """ Helper function for the path planner stream """
        if diff_norm( bgn.value, end.value ) > 0.0: 

            posnBgn, _ = row_vec_to_pb_posn_ornt( bgn.value )
            posnEnd, _ = row_vec_to_pb_posn_ornt( end.value )

            ## Sample Symbols ##
            nuSym = [bel.sample_symbol() for bel in self.beliefs]
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

        def stream_func( *args ):
            
            print( f"\nEvaluate PATH stream with args: {args}\n" )

            label, bgn, end = args
            qBgn = self.calc_ik( bgn )
            qEnd = self.calc_ik( end )
            if self.path_segment_checker( label, bgn, end ):
                yield (Path( label, bgn, end, X = [bgn, end,], Q = [qBgn, qEnd] ),)
            else:
                posnBgn, orntEnd = row_vec_to_pb_posn_ornt( bgn.value )
                posnEnd, _       = row_vec_to_pb_posn_ornt( end.value )
                posnMid = np.add( posnBgn, posnEnd ) / 2.0
                posnMid[2] = _Z_SAFE
                orntMid = orntEnd[:]
                mid      = Pose( pb_posn_ornt_to_row_vec( posnMid, orntMid ) )
                qMid    = self.calc_ik( mid )
                if self.path_segment_checker( label, bgn, mid ) and self.path_segment_checker( label, mid, end ):
                    yield (Path( label, bgn, end, X = [bgn, mid, end,], Q = [qBgn, qMid, qEnd] ),)
        return stream_func
    

    def get_stacker( self ):
        """ Return a function that computes Inverse Kinematics for a pose """

        def stream_func( *args ):
            """ A function that computes a stacking pose across 2 poses """

            print( f"\nEvaluate STACK PLACE stream with args: {args}\n" )

            labelUp, poseDn1, poseDn2 = args

            posnDn1, orntDn1 = row_vec_to_pb_posn_ornt( poseDn1.value )
            posnDn2, _       = row_vec_to_pb_posn_ornt( poseDn2.value )
            dist   = diff_norm( posnDn1, posnDn2 )
            sepMax = 2.1*_BLOCK_SCALE
            if dist <= sepMax:
                posnUp = np.add( posnDn1, posnDn2 ) / 2.0
                posnUp[2] = 2.0*_BLOCK_SCALE
                orntUp = orntDn1[:]
                poseUp = Pose( pb_posn_ornt_to_row_vec( posnUp, orntUp ) )
                print( f"STACK PLACE stream SUCCESS: {poseUp.value}\n" )
                yield (poseUp,)
            else:
                print( f"STACK PLACE stream FAILURE: {dist} > {sepMax}\n" )

        return stream_func
    

    def get_free_placement_test( self ):
        """ Return a function that checks if the pose is free from obstruction """

        def test_func( *args ):
            
            print( f"\nEvaluate PLACEMENT test with args: {args}\n" )

            # label, pose = args
            label, pose = args
            posn , _    = row_vec_to_pb_posn_ornt( pose.value )

            ## Sample Symbols ##
            # self.belief_update()
            nuSym = [bel.sample_symbol() for bel in self.beliefs]
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
        # print( dir( domain_pddl ) )

        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))

        # exit(0)

        constant_map = {}
        print( "Read files!" )

        print( 'Robot:', self.world.robot.get_name() )
        conf = Config( self.world.robot.get_joint_angles() )
        pose = Pose( pb_posn_ornt_to_row_vec( *self.world.robot.get_current_pose() ) )
        
        trgtRed = Pose( [ _MIN_X_OFFSET+2.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )
        trgtYlw = Pose( [ _MIN_X_OFFSET+4.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )
        trgtBlu = Pose( [ _MIN_X_OFFSET+6.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )
        
        tCnf = Config( [0 for _ in range(6)] )
        trgt = Pose( [ 0.492, 0.134, 0.600, 0.707, 0.0, 0.707, 0.0 ] )
        
        init = [
            ## Init Predicates ##
            ('Conf', conf),
            ('AtConf', conf),
            ('EffPose', pose),
            ('AtPose', pose),
            ('HandEmpty',),
            ## Goal Predicates ##
            # ('Conf', tCnf),
            # ('EffPose', trgt),
            ('Pose', trgtRed),
            ('Pose', trgtYlw),
            # ('Pose', trgtBlu),
        ] 
        
        for body in _ACTUAL_NAMES:
            init.append( ('Graspable', body) )

        print( f"### Initial Symbols ###" )
        for sym in init:
            print( f"\t{sym}" )

        print( "Robot grounded!" )

        # goal = ('AtConf', tCnf)  
        # goal = ('AtPose', trgt)  
        # goal = ('Holding', 'redBlock')  

        # goal = ('and',
        #         ('Obj', 'redBlock', trgtRed),
        #         # # ('Obj', 'ylwBlock', trgtYlw),
        #         # # ('Obj', 'bluBlock', trgtBlu),
        #         # ('HandEmpty',)
        # )
        
        goal = ( 'and',
            ('HandEmpty',),
            ('Obj', 'redBlock', trgtRed),
            ('Obj', 'ylwBlock', trgtYlw),
            ('Supported', 'bluBlock', 'redBlock'),
            ('Supported', 'bluBlock', 'ylwBlock'),
        ) 

        stream_map = {
            ### Symbol Streams ###
            'sample-object':      from_gen_fn( self.get_object_stream() ), 
            'sample-grasp':       from_gen_fn( self.get_grasp_stream()  ),
            'inverse-kinematics': from_gen_fn( self.get_IK_solver()     ),
            'path-planner':       from_gen_fn( self.get_path_planner()  ),
            'find-stack-place':   from_gen_fn( self.get_stacker()       ),
            ### Symbol Tests ###
            'test-free-placment': from_test( self.get_free_placement_test() ),
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

    for i in range( 10 ):
        planner.belief_update() # We need at least an initial set of beliefs in order to plan

    print( '\n\n\n##### PDLS INIT #####' )
    problem = planner.pddlstream_from_problem()
    print( 'Created!\n\n\n' )

    if 1:
        print( '##### PDLS SOLVE #####' )
        try:
            solution = solve( problem, 
                              algorithm = "adaptive", #"focused", #"binding", #"incremental", #"adaptive", 
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
        print( f"\n\n\nPlan output from PDDLStream:" )
        if plan is not None:
            for i, action in enumerate( plan ):
                print( dir( action ) )
                print( f"\t{i+1}: { action.__class__.__name__ }, {action.name}" )
                for j, arg in enumerate( action.args ):
                    print( f"\t\tArg {j}:\t{type( arg )}, {arg}" )
        else:
            print( plan )
        # print( dir( plan[0] ) )
        print( "\n\n\n" )

    if 1:
        btPlan = get_BT_plan_from_PDLS_plan( plan, world )
        print( "\n\n\n" )

        btr = BT_Runner( btPlan, world, 20.0 )
        btr.setup_BT_for_running()

        while not btr.p_ended():
            btr.tick_once()

    robot.goto_home()
    world.spin_for( 500 )

