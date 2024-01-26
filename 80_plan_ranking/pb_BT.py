########## INIT ####################################################################################

### Basic Imports ###
import builtins, datetime, time, math, sys
from time import sleep

### Special Imports ###
import numpy as np

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.composites import Sequence

### Local Imports ###
sys.path.append( "../" )
from magpie.poses import pose_error, rotate_pose

from utils import *



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
        self.logger.debug( f"[{self.name}::__init__()]" )
        if self.ctrl is None:
            self.logger.warning( f"{self.name} is NOT conntected to a robot controller!" )
        if self.world is None:
            self.logger.warning( f"{self.name} is NOT conntected to a world object!" )

        
    def setup(self):
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
builtins._DUMMYPOSE     = np.eye(4)
builtins.MP2BB = dict()  # Hack the BB object into the built-in namespace
builtins.SCAN_POSE_KEY  = "scanPoses"
builtins.HAND_OBJ_KEY   = "handHas"
PROTO_PICK_ROT = np.array( [[ 0.0,  1.0,  0.0, ],
                            [ 1.0,  0.0,  0.0, ],
                            [ 0.0,  0.0, -1.0, ]] )

### Set important BB items ###
MP2BB[ SCAN_POSE_KEY ] = dict()



########## HELPERS #################################################################################

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
            connect_BT_to_robot( child, world )

def connect_BT_to_robot_world( bt, robot, world ):
    """ Set both controller and environment for this behavior and all children """
    connect_BT_to_robot( bt, robot )
    connect_BT_to_world( bt, world )

########## MOVEMENT BEHAVIORS ######################################################################

### Constants ###
LIBBT_TS_S       = 0.25
DEFAULT_TRAN_ERR = 0.010 # 0.002
DEFAULT_ORNT_ERR = 3*np.pi/180.0



##### Move_Q #####################################

class Move_Q( BasicBehavior ):
    """ Move the joint config `qPos` """
    
    def __init__( self, qPos, name = None, ctrl = None, world = None, rotSpeed = 1.05, rotAccel = 1.4, asynch = True ):
        """ Set the target """
        # NOTE: Asynchronous motion is closest to the Behavior Tree paradigm, Avoid blocking!
        super().__init__( name, ctrl, world )
        self.qPos     = qPos
        self.rotSpeed = rotSpeed # NOT USED
        self.rotAccel = rotAccel # NOT USED
        self.asynch   = asynch # - NOT USED
    
    
    def initialise( self ):
        """ Actually Move """
        super().initialise()
        # self.ctrl.moveJ( self.qPos, self.rotSpeed, self.rotAccel, self.asynch )
        self.ctrl.set_joint_angles( self.qPos )
    
    
    def update( self ):
        """ Return SUCCESS if the target reached """
        if self.ctrl.p_moving():
            self.status = Status.RUNNING
        else:
            error = np.subtract( self.qPos, self.ctrl.get_joint_angles() )
            error = error.dot( error )
            if( error > 0.1 ):
                self.status = Status.FAILURE
            else:
                self.status = Status.SUCCESS 
        return self.status
    

##### Move_Arm ###################################
    
class Move_Arm( BasicBehavior ):
    """ Move linearly in task space to the designated pose """
    
    def __init__( self, posn, ornt, name = None, ctrl = None, world = None, linSpeed = 0.25, linAccel = 0.5, asynch = True ):
        """ Set the target """
        # NOTE: Asynchronous motion is closest to the Behavior Tree paradigm, Avoid blocking!
        super().__init__( name, ctrl, world )
        self.posn     = posn
        self.ornt     = ornt
        self.linSpeed = linSpeed # NOT USED
        self.linAccel = linAccel # NOT USED
        self.asynch   = asynch # - NOT USED
        self.epsilon  = 1e-5
        
        
    def initialise( self ):
        """ Actually Move """
        super().initialise()
        # self.ctrl.moveL( self.pose, self.linSpeed, self.linAccel, self.asynch, self.epsilon )
        self.ctrl.goto_pb_posn_ornt( self.posn, self.ornt )
        self.world.spin_for( 20 )
        
        
    def update( self ):
        """ Return true if the target reached """
        if self.ctrl.p_moving():
            self.status = Status.RUNNING
        else:
            # pM = self.ctrl.get_tcp_pose()
            posnM, orntM = self.ctrl.get_current_pose()
            pM = pb_posn_ornt_to_homog( posnM, orntM )
            pD = pb_posn_ornt_to_homog( self.posn, self.ornt )

            if 0:
                print( "src --> dst" )
                print( posnM, orntM )
                print( self.posn, self.ornt )

            [errT, errO] = pose_error( pM, pD )
            if (errT <= DEFAULT_TRAN_ERR) and (errO <= DEFAULT_ORNT_ERR):
                self.status = Status.SUCCESS
            else:
                print( self.name, ", POSE ERROR:", [errT, errO] )
                self.status = Status.FAILURE
        return self.status
    
    
##### Open_Hand ##################################
    
    
class Ungrasp( BasicBehavior ):
    """ Open fingers to max extent """
    
    def __init__( self, name = None, ctrl = None, world = None ):
        """ Set the target """
        super().__init__( name, ctrl, world )
        # self.wait_s = 0.5
        
        
    def initialise( self ):
        """ Actually Move """
        super().initialise()
        self.world.robot_release_all()
        # sleep( self.wait_s )
        
        
    def update( self ):
        """ Return true if the target reached """
        self.status = Status.SUCCESS
        return self.status
    
    
##### Close_Hand ##################################
    
    
class Grasp( BasicBehavior ):
    """ Close fingers completely """
    
    def __init__( self, objName, name = None, ctrl = None, world = None ):
        """ Set the target """
        super().__init__( name, ctrl, world )
        self.target = objName
        # self.wait_s = 0.5
                
    def initialise( self ):
        """ Actually Move """
        super().initialise()
        # self.ctrl.close_gripper()
        # sleep( self.wait_s )
        self.world.robot_grasp_block( self.target )
        
    def update( self ):
        """ Return true if the target reached """
        self.status = Status.SUCCESS
        return self.status
    
    
##### Jog_Safe ###################################

class Jog_Safe( Sequence ):
    """ Move to a target by traversing at a safe altitude """
    # NOTE: This behavior should not, on its own, assume any gripper state
    
    def __init__( self, posn, ornt, zSAFE=0.150, name="Jog_Safe", 
                  ctrl  = None, world = None ):
        """Construct the subtree"""
        super().__init__( name = name, memory = True )
        
        # Init #
        self.posn  = posn
        self.ornt  = ornt
        endPose    = pb_posn_ornt_to_homog( self.posn, self.ornt )
        self.zSAFE = max( zSAFE, endPose[2,3] ) # Eliminate (some) silly vertical movements
        self.ctrl  = ctrl
        
        # Poses to be Modified at Ticktime #
        self.targetP = endPose.copy()
        self.pose1up = _DUMMYPOSE.copy()
        self.pose2up = _DUMMYPOSE.copy()
        
        # Behaviors whose poses will be modified #
        posn, ornt = homog_to_pb_posn_ornt( self.pose1up )
        self.moveUp = Move_Arm( posn, ornt, ctrl=ctrl, world=world )
        self.moveJg = Move_Arm( posn, ornt, ctrl=ctrl, world=world )
        self.mvTrgt = Move_Arm( posn, ornt, ctrl=ctrl, world=world )
        
        
        # 1. Move direcly up from the starting pose
        self.add_child( self.moveUp )
        # 2. Translate to above the target
        self.add_child( self.moveJg )
        # 3. Move to the target pose
        self.add_child( self.mvTrgt )
       
        
    def initialise( self ):
        """
        ( Ticked first time ) or ( ticked not RUNNING ):
        Generate move waypoint, then move with condition
        """
        posn, ornt = self.ctrl.get_current_pose()
        nowPose    = pb_posn_ornt_to_homog( posn, ornt )
        
        self.pose1up = nowPose.copy()
        self.pose1up[2, 3] = self.zSAFE

        self.pose2up = self.targetP.copy()
        self.pose2up[2, 3] = self.zSAFE

        posn, ornt = homog_to_pb_posn_ornt( self.pose1up )
        # self.moveUp.pose = self.pose1up.copy()
        self.moveUp.posn = posn[:]
        self.moveUp.ornt = ornt[:]

        posn, ornt = homog_to_pb_posn_ornt( self.pose2up )
        # self.moveJg.pose = self.pose2up.copy()
        self.moveJg.posn = posn[:]
        self.moveJg.ornt = ornt[:]

        posn, ornt = homog_to_pb_posn_ornt( self.targetP )
        # self.mvTrgt.pose = self.targetP.copy()
        self.mvTrgt.posn = posn[:]
        self.mvTrgt.ornt = ornt[:]
        
        
########## MANIPULATION BEHAVIORS ##################################################################


class Pick_at_Pose( Sequence ):
    """ Grasp at a target pose (Robot Frame) while traversing at a safe altitude """
    # NOTE: This behavior should not, on its own, assume any gripper state

    def __init__( self, posn, ornt, objName, zSAFE = 0.150, name = "Pick_at_Pose", ctrl = None, world = None ):
        """Construct the subtree"""
        super().__init__( name = name, memory = 1 )
        self.ctrl  = ctrl
        self.world = world
        # 1. Open the gripper
        # if preGraspW_m is None:
        #     self.add_child(  Open_Gripper( name = "Open", ctrl = ctrl )  )
        # else:
        #     self.add_child(  Set_Gripper( preGraspW_m, name = "Open", ctrl = ctrl )  )
        self.add_child(  Ungrasp( name = "Open", ctrl = ctrl, world = world )  )
        # 2. Jog to the target
        self.add_child(  Jog_Safe( posn, ornt, zSAFE = zSAFE, name = "Jog to Grasp Pose", ctrl = ctrl, world = world )  )
        # 1. Close the gripper
        # if graspWdth_m is None:
        #     self.add_child(  Close_Hand( name = "Close", ctrl = ctrl )  )
        # else:
        #     self.add_child(  Set_Gripper( graspWdth_m, name = "Close", ctrl = ctrl )  )
        self.add_child(  Grasp( objName, name = "Close", ctrl = ctrl, world = world )  )
            
class Place_at_Pose( Sequence ):
    """ Grasp at a target pose (Robot Frame) while traversing at a safe altitude """
    # NOTE: This behavior should not, on its own, assume any gripper state

    def __init__( self, posn, ornt, zSAFE = 0.150, name = "Place_at_Pose", ctrl = None, world = None ):
        """Construct the subtree"""
        super().__init__( name = name, memory = 1 )
        self.ctrl  = ctrl
        self.world = world
        # 2. Jog to the target
        self.add_child(  Jog_Safe( posn, ornt, zSAFE = zSAFE, name = "Jog to Grasp Pose", ctrl = ctrl, world = world )  )
        # 1. Open the gripper
        # if postGraspW_m is None:
        #     self.add_child(  Open_Gripper( name = "Open", ctrl = ctrl )  )
        # else:
        #     self.add_child(  Set_Gripper( postGraspW_m, name = "Open", ctrl = ctrl )  )
        self.add_child(  Ungrasp( name = "Open", ctrl = ctrl, world = world )  )
            
########## EXECUTION ###############################################################################


class HeartRate: 
    """ Sleeps for a time such that the period between calls to sleep results in a frequency not greater than the specified 'Hz' """
    # NOTE: This fulfills a purpose similar to the rospy rate
    
    def __init__( self , Hz ):
        """ Create a rate object with a Do-Not-Exceed frequency in 'Hz' """
        self.period = 1.0 / Hz; # Set the period as the inverse of the frequency , hearbeat will not exceed 'Hz' , but can be lower
        self.last = time.time()
        
    def check_elapsed( self, reset = True ):
        """ Check if the period has elapsed, Optionally `reset` the clock """
        elapsed = time.time() - self.last
        update  = elapsed >= self.period
        if( update and reset ):
            self.last = time.time()
        return update
    
    def sleep( self ):
        """ Sleep for a time so that the frequency is not exceeded """
        elapsed = time.time() - self.last
        if elapsed < self.period:
            time.sleep( self.period - elapsed )
        self.last = time.time()


""" Return a formatted timestamp string, useful for logging and debugging """
# def nowTimeStamp(): return datetime.datetime.now().strftime(
def nowTimeStamp(): return datetime.now().strftime(
    '%Y-%m-%d_%H-%M-%S')  # http://stackoverflow.com/a/5215012/893511



class StopBeetle:
    """Invasive Beetle: Kills (stops) all branches of the tree"""

    def __init__(self, killStatus):
        """Set the status that will be assigned to all branches"""
        self.status = killStatus

    def run(self, behav):
        """Kill all subtrees"""
        for chld in behav.children:
            self.run(chld)
        behav.stop(self.status)


        
def run_BT_until_done(
    rootNode,
    N              = 10000,
    tickPause      =     0.25,
    Nverb          =    50,
    breakOnFailure = True,
    breakOnSuccess = True,
    treeUpdate     = 0,
    failureTree    = 1,
    successTree    = 0,
    world          = None
):
    """Tick root until `maxIter` is reached while printing to terminal"""

    if world is None:
        raise ValueError( "MUST provide world ref in order to run the simulation within a BT!" )

    if Nverb:
        print(
            "About to run",
            type( rootNode ),
            "named",
            rootNode.name,
            "at",
            nowTimeStamp(),
            "with",
            1 / tickPause,
            "Hz update frequency ...",
        )

    # 0. Setup
    NstepPer = int( math.ceil( tickPause / world.tIncr ) )
    pacer    = HeartRate(Hz=1 / tickPause)  # metronome
    rootNode.setup_with_descendants()

    if Nverb:
        print("Running ...\n")

    # 1. Run
    for i in range(1, N + 1):
        try:
            world.spin_for( NstepPer )
            rootNode.tick_once()
            

            if Nverb > 0 and i % Nverb == 0:
                print("\n--------- Tick {0} ---------\n".format(i))
                print("Root node, Name:", rootNode.name, ", Status:", rootNode.status)
                print("\n")
                if treeUpdate:
                    print(
                        py_trees.display.unicode_tree(root=rootNode, show_status=True)
                    )

            if breakOnFailure and (rootNode.status == Status.FAILURE):
                print("Root node", rootNode.name, "failed!\n")
                if failureTree:
                    print(
                        py_trees.display.unicode_tree(root=rootNode, show_status=True)
                    )
                break
            elif breakOnSuccess and (rootNode.status == Status.SUCCESS):
                print("Root node", rootNode.name, "succeeded!\n")
                if successTree:
                    print(
                        py_trees.display.unicode_tree(root=rootNode, show_status=True)
                    )
                break
            else:
                pacer.sleep()

        except KeyboardInterrupt:
            print("\nSimulation HALTED by user at", nowTimeStamp())
            break

    print("\nRun completed! with status:", rootNode.status, "\n\n")

    insect = StopBeetle(rootNode.status)

    for i in range(3):
        rootNode.visit(insect)  # HACK required coz tree doesn't complete sometimes
        sleep(0.5)

    print("Root node", rootNode.name, "was killed by the running script!")