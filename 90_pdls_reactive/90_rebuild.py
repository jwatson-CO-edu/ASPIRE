########## INIT ####################################################################################

##### Imports #####

### Standard ###
import sys, time, datetime, pickle, math
now = time.time
from random import random
from collections import Counter
from itertools import count

### Special ###
import numpy as np
from scipy.stats import chi2

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
from magpie.poses import translation_diff

from utils import ( row_vec_to_homog, row_vec_to_pb_posn_ornt, roll_outcome, get_confusion_matx, 
                    multiclass_Bayesian_belief_update, pb_posn_ornt_to_row_vec, closest_dist_Q_to_segment_AB,
                    diff_norm, )
# from beliefs import ObjectSymbol
from env_config import ( _ACCEPT_POSN_ERR, _GRASP_VERT_OFFSET, _Z_SAFE, _GRASP_ORNT_XYZW, _NULL_NAME, 
                         _NULL_THRESH, _POSN_STDDEV, _BLOCK_NAMES, _N_POSE_UPDT, _SUPPORT_NAME, _BLOCK_SCALE, )
from pb_BT import Pick_at_Pose, Place_at_Pose, connect_BT_to_robot_world

########## DOMAIN OBJECTS ##########################################################################

class Pose: 
    """ A named object, it's pose, and the surface that supports it """
    num = count()
    def __init__( self, ref, label, pose, surf = _SUPPORT_NAME ):
        self.ref    = ref # - Belief from which this symbols was sampled
        self.label  = label # Sampled object label
        self.pose   = pose #- Sampled object pose
        self.action = None #- Action to which this symbol was assigned
        self.surf   = surf # WARNING: I do not like that this is part of the predicate!
        self.index  = next( self.num )

    def prob( self ):
        """ Get the current belief this symbol is true based on the belief this symbol was drawn from """
        return self.ref.labels[self.label]
    
    def __repr__( self ):
        """ String representation, Including current symbol belief """
        return f"<{self.label} @ {self.pose}, P={self.prob() if (self.ref is not None) else None}>"
    
    def p_attached( self ):
        """ Return true if this symbol has been assigned to an action """
        return (self.action is not None)
    
    @property
    def value( self ):
        return self.pose


class ObjectBelief:
    """ Hybrid belief: A discrete distribution of classes that may exist at a continuous distribution of poses """

    def reset_covar( self ):
        self.covar   = np.zeros( (7,7,) ) # ------ Pose covariance matrix
        for i, stdDev in enumerate( self.pStdDev ):
            self.covar[i,i] = stdDev * stdDev

    def __init__( self, initStddev = _POSN_STDDEV ):
        """ Initialize with origin poses and uniform, independent variance """
        # stdDev = [initStddev if (i<3) else 0.0 for i in range(7)]
        stdDev = [initStddev for i in range(7)]
        self.labels  = {} # ---------------------- Current belief in each class
        self.pose    = np.array([0,0,0,1,0,0,0]) # Mean pose
        self.pStdDev = np.array(stdDev) # -------- Pose variance
        self.pHist   = [] # ---------------------- Recent history of poses
        self.pThresh = 0.5 # --------------------- Minimum prob density at which a nearby pose is relevant
        self.reset_covar()
        self.visited = False

    def get_posn( self, poseOrBelief ):
        """ Get the position from the object """
        if isinstance( poseOrBelief, (ObjectBelief, Pose) ):
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
            support    = self.object_supporting_pose( poseSample )
        except np.linalg.LinAlgError:
            self.reset_covar()
            poseSample = np.random.multivariate_normal( self.pose, self.covar ) 
        return Pose( 
            self,
            label, 
            poseSample,
            support
        )
    
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
        poseHist   = np.array( self.pHist )
        self.pHist = []
        nuPose     = np.mean( poseHist, axis = 0 )
        nuStdDev   = np.std( poseHist, axis = 0 )
        nuvar      = np.zeros( (7,7,) ) # ------ Pose covariance matrix
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
        nuSum = np.add( 
            np.reciprocal( self.covar, where = self.covar != 0.0 ), 
            np.reciprocal( nuvar, where = nuvar != 0.0 ) 
        )
        self.covar = np.reciprocal( nuSum, where = nuSum != 0.0 )
    
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


    
class BodyConf:
    """ A robot configuration """
    num = count()
    def __init__( self, name, config ):
        self.name = name
        self.cnfg = config
        self.index = next( self.num )
    def __repr__( self ):
        return f"<Config: {self.name}, {self.index}>"
    @property
    def value( self ):
        return self.cnfg
    @property # WARNING: IS THIS REDUNDANT?
    def values( self ):
        return self.cnfg
    
class BodyGrasp:
    """ Enough info to pick up something """
    num = count()
    def __init__( self, body, grasp_pose, approach_pose ):
        self.body          = body
        self.grasp_pose    = grasp_pose
        self.approach_pose = approach_pose
        self.index = next( self.num )
    def __repr__( self ):
        # return f"<Grasp: {self.body}, [{self.grasp_pose[0,3]}, {self.grasp_pose[1,3]}, {self.grasp_pose[2,3]}]>"
        return f"<Grasp: {self.body}, {self.index}>"
    @property
    def value( self ):
        return self.grasp_pose
    
class BodyPath:
    """ Path of something between two configs """
    num = count()
    def __init__( self, body, path ):
        self.body = body
        self.path = path[:]
        self.index = next( self.num )
    def __repr__( self ):
        wpStr = "["
        for wp in self.path:
            wpStr += str( wp ) + ","
        wpStr += "]"
        return f"<Trajectory: {self.body}, {self.index}, {wpStr}>"
    @property
    def value( self ):
        return self.path

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

# FIXME: NEED TO SEPARATE PICK AND PLACE

class Pick_and_Place( Sequence ):
    """ BT that produces <OBJ@LOC> """

    def __init__( self, objName, dest, world = None, robot = None ):
        super().__init__( name = f"Pick_and_Place: {objName} --> {dest}" )
        self.objName = objName # Type of object required
        self.dest    = dest # -- Destination pose
        self.symbol  = None # -- Symbol on which this behavior relies
        self.msg     = "" # ---- Message: Reason this action failed -or- OTHER
        self.ctrl    = robot # - Agent that executes
        self.world   = world  #- Simulation ref

    def __repr__( self ):
        """ String representation of the action """
        return f"Pick_and_Place: {self.objName} --> {self.dest}"
    
    def get_grounded( self, symbol ):
        """ Copy action with a symbol attached """
        rtnAct = Pick_and_Place( self.objName, self.dest, self.world, self.ctrl )
        rtnAct.symbol = symbol
        symbol.action = rtnAct
        return rtnAct

    def copy( self ):
        """ Deep copy """
        rtnObj = Pick_and_Place( self.objName, self.dest, self.world, self.ctrl )
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
        """ Get the linear distance between the symbol pose and the destination """
        return translation_diff( row_vec_to_homog( self.dest ), row_vec_to_homog( self.symbol.pose ) )

    def prep( self ):
        """ Use the symbol grounding to parameterize the BT """

        # 0. Check if the goal has already been met, PDLS FIXME
        miniGoal = Pose( None, self.objName, self.dest, _SUPPORT_NAME )
        if self.world.check_predicate( miniGoal, _ACCEPT_POSN_ERR ):
            print( f"{self.name} ALREADY DONE" )
            self.status = Status.SUCCESS
            return

        # 1. Fetch ref to the object nearest the pose, if any
        posnEnd, orntEnd = row_vec_to_pb_posn_ornt( self.dest )
        graspPose = self.symbol.pose
        handle    = self.world.get_handle_at_pose( graspPose )

        if handle is not None:
            targetNam = self.world.get_handle_name( handle )

            posnTgt, orntTgt = row_vec_to_pb_posn_ornt( graspPose )
            
            posnTgt[2] += _GRASP_VERT_OFFSET
            orntTgt = _GRASP_ORNT_XYZW.copy()
            
            posnEnd[2] += _GRASP_VERT_OFFSET
            orntEnd = orntTgt[:]

            self.add_child( Pick_at_Pose( posnTgt, orntTgt, targetNam, zSAFE = _Z_SAFE, name = "Pick_at_Pose", 
                                          ctrl = self.ctrl, world = self.world ) )
            self.add_child( Place_at_Pose( posnEnd, orntEnd, zSAFE = _Z_SAFE, name = "Place_at_Pose", 
                                           ctrl = self.ctrl, world = self.world ) )
        else:
            self.status = Status.FAILURE
            self.msg    = "Object miss"
        


########## PLANNER HELPERS #########################################################################

def prep_plan( plan ):
    """ Set appropriate targets """
    for action in plan:
        action.prep_action()

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
        rtnGoal = []
        for action in self:
            rtnGoal.append( Pose( None, action.objName, action.dest, _SUPPORT_NAME ) )
        return rtnGoal
    


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
        self.world   = world
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

    def get_pose_stream( self ):
        """ Return a function that returns poses """

        def stream_func( *args ):
            """ A function that returns poses """
            objName, spprtName = args

            ## Sample Symbols ##
            nuSym = [bel.sample_symbol() for bel in self.beliefs]
            for sym in nuSym:
                if (sym.surf == spprtName) and (objName == sym.name):
                    yield sym
            # else yield nothing if we cannot certify the object!

        return stream_func

    def get_grasp_stream( self ):
        """ Return a function that returns grasps """
        
        def stream_func( *args ):
            """ A function that returns grasps """
            objName = args[0]
            objPose = args[1].pose

            if objPose is not None:
                grasp_pose = objPose[:]
                grasp_pose[2] += _GRASP_VERT_OFFSET
                posn, _ = row_vec_to_pb_posn_ornt( grasp_pose )
                ornt = _GRASP_ORNT_XYZW.copy()
                grasp_pose = pb_posn_ornt_to_row_vec( posn, ornt )

                posn[2] = _Z_SAFE
                approach_pose = pb_posn_ornt_to_row_vec( posn, ornt )

                grasp = BodyGrasp( objName, grasp_pose, approach_pose )
                print( "Got a grasp!\n", grasp_pose )
                yield (grasp,)
             # else yield nothing if we cannot certify the object!

        return stream_func

    def get_free_motion_planner( self ):
        """ Return a function that checks if the path is free """

        def stream_func( *args ):
            """ A function that checks if the path is free """

            (bgn, end) = args
            if bgn != end:

                posnBgn, _ = row_vec_to_pb_posn_ornt( bgn )
                posnEnd, _ = row_vec_to_pb_posn_ornt( end )

                objs = self.world.full_scan_true()
                okay = True
                for sym in objs:
                    Q, _ = row_vec_to_pb_posn_ornt( sym.pose )
                    d = closest_dist_Q_to_segment_AB( Q, posnBgn, posnEnd )
                    if d < 2.0*_BLOCK_SCALE:
                        okay = False
                        break
                if okay:
                    yield (BodyPath( self.world.robotName, [bgn, end] ),) 

        return stream_func

    def get_IK_solver( self ):
        """ Return a function that computes Inverse Kinematics for a pose """

        def stream_func( *args ):
            """ A function that computes Inverse Kinematics for a pose """

            (trgt, pose, grasp) = args

            posn, ornt = row_vec_to_pb_posn_ornt( grasp.approach_pose )
            apprcQ = self.world.robot.calculate_ik_quat( posn, ornt )
            posn, ornt = row_vec_to_pb_posn_ornt( grasp.grasp_pose )
            graspQ = self.world.robot.calculate_ik_quat( posn, ornt )

            #     ( <IK Sol'n>, <Trajectory> )
            yield ( BodyConf( self.world.robotName, graspQ ), 
                    BodyPath( self.world.robotName, [
                        BodyConf( self.world.robotName, apprcQ ),
                        BodyConf( self.world.robotName, graspQ ),
                    ] ) )
            # FIXME: DO NOT YIELD OF THE POSE CANNOT BE REACHED!
            
        return stream_func

    def get_safe_pose_test( self ):
        """ Return a function that returns True if two objects are a safe distance apart """

        def test( *args ):
            """ Do not pass if it is too close to other blocks """
            label1, pose1, label2, pose2 = args
            if diff_norm( pose1[:3], pose2[:3] ) <= 2.0*_BLOCK_SCALE:
                return False
            else:
                return True
            
        return test

    def get_arch_stream( self ):
        """ Return a stream that samples arch locations """

        # FIXME, START HERE: WHEN TO YIELD THIS? WHAT DOES IT MEAN?

        def stream_func( *args ):
            """ A function that returns poses """
            objName, spprt1, spprt2 = args

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

#             objEvidence = self.world.full_scan_noisy()
#             print( f"{i+1}: Got {len(objEvidence)} beliefs!" )

#             ## Integrate Beliefs ##
#             cNu = 0
#             cIn = 0
#             self.unvisit_beliefs()
#             for objEv in objEvidence:
#                 if self.integrate_one_segmentation( objEv ):
#                     cIn += 1
#                 else:
#                     cNu += 1
#             if (cNu or cIn):
#                 print( f"\t{cNu} new object beliefs this iteration!" )
#                 print( f"\t{cIn} object beliefs updated!" )
#             else:
#                 print( f"\tNO belief update!" )
#             print( f"Total Beliefs: {len(self.beliefs)}" )
            
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
        

# ########## MAIN ####################################################################################

# ##### Env. Settings #####
# np.set_printoptions( precision = 3, linewidth = 145 )


# ##### Run Sim #####
# if __name__ == "__main__":

#     ### Init ###

#     world   = PB_BlocksWorld()
#     robot = world.robot
#     robot.goto_home()
#     world.spin_for( 500 )

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
