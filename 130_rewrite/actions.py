########## INIT ####################################################################################

import math
from random import random

import py_trees
from py_trees.common import Status
from py_trees.composites import Sequence

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
        self.status = Status.FAILURE
        self.msg    = msg
        self.world.robot_release_all()
        self.world.spin_for( 250 )

    def tick_once( self ):
        """ Run one simulation step """
        self.world.spin_for( self.Nstep )
        if hasattr( self.root, "least_prob_symbol" ) and ( self.root.least_prob_symbol() <  _PLAN_THRESH):
            self.set_fail( "CONFUSION ERROR" )
        if not self.p_ended():
            self.root.tick_once()
        self.status = self.root.status
        self.i += 1
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

    def __init__( self, args = None, goal = None, world = None, robot = None, name = "Grounded Sequence" ):
        """ Init BT """
        super().__init__( name = name, memory = True )
        self.args    = args if (args is not None) else list() # Symbols involved in this action
        self.symbols = list() # -- Symbol on which this behavior relies
        self.msg     = "" # ---- Message: Reason this action failed -or- OTHER
        self.ctrl    = robot # - Agent that executes
        self.world   = world  #- Simulation ref
    
    def __repr__( self ):
        """ Get the name, Assume child classes made it sufficiently descriptive """
        return str( self.name )

########## PDLS --TO-> BT ##########################################################################

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