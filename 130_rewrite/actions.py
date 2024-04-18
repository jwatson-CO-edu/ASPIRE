########## INIT ####################################################################################

##### Imports #####

### Standard ###
import math, sys
from random import random

### Special ###
import numpy as np

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.composites import Sequence

### Local ###
from utils import row_vec_to_homog
from symbols import extract_row_vec_pose
from env_config import _Z_SAFE

sys.path.append( "../" )
from magpie.poses import pose_error

##### Constants #####

_HAND_WAIT   = 100
_GRASP_PAUSE = False



########## BEHAVIOR HELPERS ########################################################################

def pass_msg_up( bt, failBelow = False ):
    if bt.parent is not None:
        if bt.status == Status.FAILURE:
            if (bt.parent.status != Status.FAILURE) or (len( bt.parent.msg ) == 0):
                setattr( bt.parent, "msg", bt.msg )
                pass_msg_up( bt.parent, True )
            else:
                pass_msg_up( bt.parent )
        elif failBelow:
            setattr( bt.parent, "msg", bt.msg )
            pass_msg_up( bt.parent, True )


def connect_BT_to_robot( bt, robot ):
    """ Assign `robot` controller to `bt` and recursively to all nodes below """
    if hasattr( bt, 'ctrl' ):
        bt.ctrl = robot
    if len( bt.children ):
        for child in bt.children:
            connect_BT_to_robot( child, robot )


def connect_BT_to_world( bt, world ):
    """ Assign `world` environment to `bt` and recursively to all nodes below """
    if hasattr( bt, 'world' ):
        bt.world = world
    if len( bt.children ):
        for child in bt.children:
            connect_BT_to_world( child, world )


def connect_BT_to_robot_world( bt, robot, world ):
    """ Set both controller and environment for this behavior and all children """
    connect_BT_to_robot( bt, robot )
    connect_BT_to_world( bt, world )



########## BASE CLASS ##############################################################################

class BasicBehavior( Behaviour ):
    """ Abstract class for repetitive housekeeping """
    
    def __init__( self, name = None, ctrl = None, world = None ):
        """ Set name to the child class name unless otherwise specified """
        if name is None:
            super().__init__( name = str( self.__class__.__name__  ) )
        else:
            super().__init__( name = name )
        self.ctrl  = ctrl
        self.world = world
        self.msg   = ""
        self.logger.debug( f"[{self.name}::__init__()]" )
        if self.ctrl is None:
            self.logger.warning( f"{self.name} is NOT conntected to a robot controller!" )
        if self.world is None:
            self.logger.warning( f"{self.name} is NOT conntected to a world object!" )
        

    def setup(self):
        """ Virtual setup for base class """
        self.logger.debug( f"[{self.name}::setup()]" )          
        
        
    def initialise( self ):
        """ Run first time behaviour is ticked or not RUNNING.  Will be run again after SUCCESS/FAILURE. """
        self.status = Status.RUNNING # Do not let the behavior idle in INVALID
        self.logger.debug( f"[{self.name}::initialise()]" )          

        
    def terminate( self, new_status ):
        """ Log how the behavior terminated """
        self.status = new_status
        self.logger.debug( f"[{self.name}::terminate().terminate()][{self.status}->{new_status}]" )
        
        
    def update( self ):
        """ Return true in all cases """
        self.status = py_trees.common.Status.SUCCESS
        return self.status
    
    
    
########## CONSTANTS & COMPONENTS ##################################################################

### Init data structs & Keys ###
_DUMMYPOSE     = np.eye(4)
MP2BB          = dict()  # Hack the BB object into the built-in namespace
SCAN_POSE_KEY  = "scanPoses"
HAND_OBJ_KEY   = "handHas"
PROTO_PICK_ROT = np.array( [[ 0.0,  1.0,  0.0, ],
                            [ 1.0,  0.0,  0.0, ],
                            [ 0.0,  0.0, -1.0, ]] )

### Set important BB items ###
MP2BB[ SCAN_POSE_KEY ] = dict()



########## BASIC BEHAVIORS #########################################################################

### Constants ###
LIBBT_TS_S       = 0.25
DEFAULT_TRAN_ERR = 0.010 # 0.002
DEFAULT_ORNT_ERR = 3*np.pi/180.0



##### Move_Effector ###################################
    
class Move_Effector( BasicBehavior ):
    """ Move linearly in task space to the designated pose """
    
    def __init__( self, pose, name = None, ctrl = None, world = None, linSpeed = None ):
        """ Set the target """
        # NOTE: Asynchronous motion is closest to the Behavior Tree paradigm, Avoid blocking!
        super().__init__( name, ctrl, world )
        self.pose     = pose
        self.linSpeed = linSpeed 
        self.epsilon  = 1e-5
        
        
    def initialise( self ):
        """ Actually Move """
        super().initialise()
        # self.ctrl.goto_pb_posn_ornt( self.posn, self.ornt )
        if self.linSpeed is not None:
            self.ctrl.set_step_speed( self.linSpeed )
        self.ctrl.goto_pose( self.pose )
        
        
    def update( self ):
        """ Return true if the target reached """

        if self.ctrl.p_moving():
            self.status = Status.RUNNING
        else:
            hmgR = row_vec_to_homog( self.ctrl.get_current_pose() )
            hmgT = row_vec_to_homog( self.pose )
            
            print( hmgR, hmgT )

            [errT, errO] = pose_error( hmgR, hmgT )
            if (errT <= DEFAULT_TRAN_ERR) and (errO <= DEFAULT_ORNT_ERR):
                self.status = Status.SUCCESS
            else:
                print( self.name, ", POSE ERROR:", [errT, errO] )
                self.msg = "Move_Arm: POSE ERROR"
                pass_msg_up( self )
                self.status = Status.FAILURE

        return self.status



##### Grasp ###########################################
    
class Grasp( BasicBehavior ):
    """ Close fingers completely """
    
    def __init__( self, objName, objPose = None, name = None, ctrl = None, world = None ):
        """ Set the target """
        super().__init__( name, ctrl, world )
        self.target  = objName
        self.objPose = objPose
                
    def initialise( self ):
        """ Actually Move """
        super().initialise()
        if _GRASP_PAUSE:
            self.world.spin_for( _HAND_WAIT )
        self.world.robot_grasp_at( self.objPose )
        if _GRASP_PAUSE:
            self.world.spin_for( _HAND_WAIT )
        
    def update( self ):
        """ Return true if the target reached """
        self.status = Status.SUCCESS
        return self.status
    


##### Ungrasp #########################################
    
class Ungrasp( BasicBehavior ):
    """ Open fingers to max extent """
    
    def __init__( self, name = None, ctrl = None, world = None ):
        """ Set the target """
        super().__init__( name, ctrl, world )
        
        
    def initialise( self ):
        """ Actually Move """
        super().initialise()
        if _GRASP_PAUSE:
            self.world.spin_for( _HAND_WAIT )
            self.world.robot_release_all()
            self.world.spin_for( 2 )
        self.world.robot_release_all()
        if _GRASP_PAUSE:
            self.world.spin_for( _HAND_WAIT )
        
        
    def update( self ):
        """ Return true if the target reached """
        self.status = Status.SUCCESS
        return self.status
    


########## HELPER FUNCTIONS ########################################################################

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



########## BT-PLANNER INTERFACE ####################################################################

class BT_Runner:
    """ Run a BT with checks """

    def __init__( self, root, world, tickHz = 4.0, limit_s = 20.0 ):
        """ Set root node and world reference """
        self.root   = root
        self.world  = world
        self.status = Status.INVALID
        self.freq   = tickHz
        self.Nstep  = int( max(1.0, math.ceil((1.0 / tickHz) / world.period)))
        self.msg    = ""
        self.Nlim   = int( limit_s * tickHz )
        self.i      = 0


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
    

    def set_fail( self, msg = "DEFAULT MSG: STOPPED" ):
        """ Handle external signals to halt BT execution """
        self.status = Status.FAILURE
        self.msg    = msg
        self.world.robot_release_all()
        self.world.spin_for( 250 )


    def tick_once( self ):
        """ Run one simulation step """
        ## Let sim run ##
        self.world.spin_for( self.Nstep )
        ## Advance BT ##
        if not self.p_ended():
            self.root.tick_once()
        self.status = self.root.status
        self.i += 1
        ## Check Conditions ##
        if (self.i >= self.Nlim) and (not self.p_ended()):
            self.set_fail( "BT TIMEOUT" )
        if self.p_ended():
            pass_msg_up( self.root )
            if len( self.msg ) == 0:
                self.msg = self.root.msg
            self.display_BT() 



########## BLOCKS DOMAIN BEHAVIOR TREES ############################################################

class GroundedAction( Sequence ):
    """ This is the parent class for all actions available to the planner """

    def __init__( self, args = None, world = None, robot = None, name = "Grounded Sequence" ):
        """ Init BT """
        super().__init__( name = name, memory = True )
        self.args    = args if (args is not None) else list() # Symbols involved in this action
        self.symbols = list() #  Symbol on which this behavior relies
        self.msg     = "" # ---- Message: Reason this action failed -or- OTHER
        self.ctrl    = robot # - Agent that executes
        self.world   = world  #- Simulation ref
    
    def __repr__( self ):
        """ Get the name, Assume child classes made it sufficiently descriptive """
        return str( self.name )
    


class MoveFree( GroundedAction ):
    """ Move the unburdened effector to the given location """
    def __init__( self, args, world = None, robot = None, name = None ):

        # ?poseBgn ?poseEnd
        poseBgn, poseEnd = args

        if name is None:
            name = f"Move Free --to-> {poseEnd.pose}"

        super().__init__( args, world, robot, name )

        poseEnd = extract_row_vec_pose( poseEnd )
                
        self.add_child(
            Move_Effector( poseEnd, ctrl = robot, world = world )
        )



class Pick( GroundedAction ):
    """ Add object to the gripper payload """
    def __init__( self, args, goal = None, world = None, robot = None, name = None ):

        # ?label ?pose ?prevSupport
        label, pose, prevSupport = args
        
        if name is None:
            name = f"Pick {label} at {pose.pose} from {prevSupport}"
        super().__init__( args, world, robot, name )

        self.add_child( 
            Grasp( label, pose, name = name, ctrl = robot, world = world )
        )



class MoveHolding( GroundedAction ):
    """ Move the burdened effector to the given location """
    def __init__( self, args, world = None, robot = None, name = None ):

        # ?poseBgn ?poseEnd ?label
        poseBgn, poseEnd, label = args

        if name is None:
            name = f"Move Holding {label} --to-> {poseEnd.pose}"
        super().__init__( args, world, robot, name )

        poseBgn = extract_row_vec_pose( poseBgn )
        poseEnd = extract_row_vec_pose( poseEnd )
        posnBgn = poseBgn[:3]
        posnEnd = poseEnd[:3]
        posnMid = np.add( posnBgn, posnEnd ) / 2.0
        posnMid[2] = _Z_SAFE
        poseMid = posnMid.tolist() + [1,0,0,0]
                
        self.add_children( [
            Move_Effector( poseMid, ctrl = robot, world = world ),
            Move_Effector( poseEnd, ctrl = robot, world = world ),
        ] )



class Stack( GroundedAction ):
    """ Let go of gripper payload """
    def __init__( self, args, world = None, robot = None, name = None ):

        # ?labelUp ?poseUp ?labelDn
        labelUp, poseUp, labelDn = args
        
        if name is None:
            name = f"Stack {labelUp} on top of {labelDn} at {poseUp.pose}"
        super().__init__( args, world, robot, name )

        self.add_child( 
            Ungrasp( name = name, ctrl = robot, world = world )
        )



class Place( GroundedAction ):
    """ Let go of gripper payload """
    def __init__( self, args, world = None, robot = None, name = None ):

        # ?label ?pose
        label, pose = args
        
        if name is None:
            name = f"Place {label} at {pose.pose}"
        super().__init__( args, world, robot, name )

        self.add_child( 
            Ungrasp( name = name, ctrl = robot, world = world )
        )



########## PLANS ###################################################################################

class Plan( Sequence ):
    """ Special BT `Sequence` with assigned priority, cost, and confidence """

    def __init__( self ):
        """ Set default priority """
        super().__init__( name = "PDDLStream Plan", memory = True )
        self.msg    = "" # --------------- Message: Reason this plan failed -or- OTHER
        self.ctrl   = None
        self.world  = None
    
    def __len__( self ):
        """ Return the number of children """
        return len( self.children )

    def append( self, action ):
        """ Add an action """
        self.add_child( action )
    
    def __repr__( self ):
        """ String representation of the plan """
        return f"<{self.name}, Status: {self.status}>"



########## PDLS --TO-> BT ##########################################################################

def get_ith_BT_action_from_PDLS_plan( pdlsPlan, i, world ):
    actName  = pdlsPlan[i].name
    actArgs  = pdlsPlan[i].args
    btAction = None
    if actName == "move_free":
        btAction = MoveFree( actArgs, world = world, robot = world.robot )
    elif actName == "pick":
        btAction = Pick( actArgs, world = world, robot = world.robot )
    elif actName == "move_holding":
        btAction = MoveHolding( actArgs, world = world, robot = world.robot )
    elif actName == "place":
        btAction = Place( actArgs, world = world, robot = world.robot )
    elif actName == "stack":
        btAction = Stack( actArgs, world = world, robot = world.robot )
    else:
        raise NotImplementedError( f"There is no BT procedure defined for a PDDL action named {actName}!" )
    print( f"Action {i+1}, {actName} --> {btAction.name}, planned!" )
    return btAction


def get_BT_plan_until_block_change( pdlsPlan, world ):
    """ Translate the PDLS plan to one that can be executed by the robot """
    rtnBTlst = []
    if pdlsPlan is not None:
        for i in range( len( pdlsPlan ) ):
            btAction = get_ith_BT_action_from_PDLS_plan( pdlsPlan, i, world )
            rtnBTlst.append( btAction )
            if btAction.__class__ in ( Place, Stack ):
                break

    rtnPlan = Plan()
    rtnPlan.add_children( rtnBTlst )
    return rtnPlan