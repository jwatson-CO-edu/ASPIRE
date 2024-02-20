import math
from random import random

import py_trees
from py_trees.common import Status
from py_trees.composites import Sequence

from pb_BT import Move_Arm, Grasp, Ungrasp, connect_BT_to_robot_world, pass_msg_up
from utils import row_vec_to_pb_posn_ornt

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
    
        for x_i in traj.wp[1:]:
            grasp_pose = list( x_i.grasp )
            posn, ornt = row_vec_to_pb_posn_ornt( grasp_pose )
            self.add_child( 
                Move_Arm( posn, ornt, name = name, ctrl = robot, world = world )
            )


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


def get_ith_BT_action_from_PDLS_plan( pdlsPlan, i, world ):
    actName  = pdlsPlan[i].name
    actArgs  = pdlsPlan[i].args
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
    return btAction


def get_BT_plan_from_PDLS_plan( pdlsPlan, world ):
    """ Translate the PDLS plan to one that can be executed by the robot """
    rtnBTlst = []
    if pdlsPlan is not None:
        # for i, pdlsAction in enumerate( pdlsPlan ):
        for i in range( len( pdlsPlan ) ):
            btAction = get_ith_BT_action_from_PDLS_plan( pdlsPlan, i, world )
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

