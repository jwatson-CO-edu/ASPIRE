########## INIT ####################################################################################

##### Imports #####
import sys, pickle
from collections import Counter
from math import log, isnan

sys.path.append( "../" )
from magpie.poses import translation_diff

from utils import *
from pb_BT import *


##### Paths #####
ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
TABLE_URDF_PATH = os.path.join( pybullet_data.getDataPath(), "table/table.urdf" )

##### Constants #####
_BLOCK_SCALE  = 0.038
_MIN_X_OFFSET = 0.400



########## DEV PLAN ################################################################################
"""
[>] Pick and Place behavior
"""


def make_table():
    """ Load a table """
    # table = pb.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
    return pb.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])

def rand_table_pose():
    """ Return a random pose in the direct viscinity if the robot """
    return [ 
        _MIN_X_OFFSET + random()*8.0*_BLOCK_SCALE, 
        random()*16.0*_BLOCK_SCALE-8.0*_BLOCK_SCALE, 
        _BLOCK_SCALE 
    ], [0, 0, 0, 1]

def make_block():
    """ Load a block at the correct scale, place it random, and return the int handle """
    posn, _ = rand_table_pose()
    return pb.loadURDF( 
        "cube.urdf", 
        posn,
        globalScaling = 0.25/4.0
    )


########## UR5 ROBOT ###############################################################################

_GRASP_VERT_OFFSET = _BLOCK_SCALE * 2.0
_GRASP_ORNT_XYZW   = np.array( [0, 0.7070727, 0, 0.7071408,] )
_Q_HOME            = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]
_ROT_VEL_SMALL     = 0.005

class UR5Sim:
    """ Original Author: Josep Daniel, https://github.com/josepdaniel/ur5-bullet/tree/master/UR5 """

    def load_robot( self ):
        """ Load UR5 from description """
        flags = 0 #pb.URDF_USE_SELF_COLLISION
        robot = pb.loadURDF( ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags )
        return robot
  
    def __init__( self ):
        """ Load robot and controller """
        
        self.end_effector_index = 7 # 5 # 6 # 7
        self.jntIndices = [1,2,3,4,5,6]
        self.ur5 = self.load_robot()
        self.num_joints = pb.getNumJoints( self.ur5 )
        
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range( self.num_joints ):
            info = pb.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pb.setJointMotorControl2(self.ur5, info.id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info     

        self.Nfails =  0
        self.Nlimit =  3
        self.lastT  = self.get_current_pose()

    def set_joint_angles( self, joint_angles ):
        """ Set a joint-space goal and pursue it with maximum force """
        poses   = []
        indexes = []
        forces  = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pb.setJointMotorControlArray(
            self.ur5, indexes,
            pb.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            # positionGains=[0.04]*len(poses), forces=forces
            positionGains=[0.06]*len(poses), forces=forces
        )

    def get_joint_vel( self ):
        """ Return the current joint velocities """
        j = pb.getJointStates( self.ur5, self.jntIndices )
        return [i[1] for i in j]
    
    # def get_joint_acc( self ):
    #     """ Return the current joint velocities """
    #     j = pb.getJointStates( self.ur5, self.jntIndices )
    #     return [i[2] for i in j]
    
    def p_moving( self ):
        """ Return True if any of the joint velocities are above some small number """
        vel = self.get_joint_vel()
        print( f"Joint Vel: {vel}" )
        for v in vel:
            if abs( v ) > _ROT_VEL_SMALL:
                return True
        return False

    def get_joint_angles( self ):
        """ Return the current joint configuration """
        j = pb.getJointStates( self.ur5, self.jntIndices )
        return [i[0] for i in j]
    

    def check_collisions( self, verbose = False ):
        """ Return True if there is a self-collision """
        collisions = pb.getContactPoints()
        if len(collisions) > 0:
            if verbose:
                print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False


    def calculate_ik_euler( self, position, orientation ):
        """ Get the target joint angles to achieve the desired `position` and `orientation` (Euler Angles) """
        quaternion   = pb.getQuaternionFromEuler(orientation)
        lower_limits = [-2*math.pi]*6
        upper_limits = [2*math.pi]*6
        joint_ranges = [4*math.pi]*6
        rest_poses   = _Q_HOME
        joint_angles = pb.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
    

    def calculate_ik_quat( self, position, quaternion ):
        """ Get the target joint angles to achieve the desired `position` and `orientation` (Quaternion) """
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        # rest_poses   = _Q_HOME
        rest_poses   = self.get_joint_angles()
        joint_angles = pb.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.1]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses,
            maxNumIterations  = 2000,
            residualThreshold = 0.005
        )
        return joint_angles
    
    def goto_home( self ):
        """ Go to the home config """
        self.set_joint_angles( _Q_HOME )

    def goto_pb_posn_ornt( self, posn, ornt ):
        """ Set the target joint angles to achieve the desired `position` and `orientation` (Quaternion) """
        q = self.calculate_ik_quat( posn, ornt )
        self.set_joint_angles( q )
        
    def get_current_pose( self ):
        """ Get the current pose in the lab frame """
        linkstate = pb.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = list(linkstate[0]), list(linkstate[1])
        return position, orientation
    
    def anger( self, posn, ornt ):
        scl = 2.0
        posn = np.add( posn, [ 
            _MIN_X_OFFSET + random()*scl*_BLOCK_SCALE, 
            random()*2.0*scl*_BLOCK_SCALE-scl*_BLOCK_SCALE, 
            0.5*_BLOCK_SCALE 
        ] )
        self.goto_pb_posn_ornt( posn, ornt )

    def log_fail( self, posn, ornt ):
        self.Nfails += 1
        if self.Nfails >= self.Nlimit:
            self.Nfails = 0
            self.anger( posn, ornt )
        

########## ENVIRONMENT #############################################################################
_POSN_STDDEV = 0.008
_NULL_NAME   = "NOTHING"
_NULL_THRESH = 0.75
_N_POSE_UPDT = 25
_BLOCK_NAMES = ['redBlock', 'ylwBlock', 'bluBlock', 'grnBlock', 'ornBlock', 'vioBlock', _NULL_NAME]

class DummyBelief:
    """ Stand-in for an actual `ObjectBelief` """
    def __init__( self, label ):
        self.labels = { label: 1.0 }



class PB_BlocksWorld:
    """ Simple physics simulation with blocks """

    def __init__( self ):
        """ Create objects """
        ## Init Sim ##
        self.tIncr         = 1.0 / 240.0
        self.physicsClient = pb.connect( pb.GUI ) # or p.DIRECT for non-graphical version
        pb.setAdditionalSearchPath( pybullet_data.getDataPath() ) #optionally
        pb.setGravity( 0, 0, -10 )

        ## Instantiate Robot and Table ##
        self.table = make_table()
        self.robot = UR5Sim()
        self.grasp = []

        ## Instantiate Blocks ##
        redBlock = make_block()
        pb.changeVisualShape( redBlock, -1, rgbaColor=[1.0, 0.0, 0.0, 1] )

        ylwBlock = make_block()
        pb.changeVisualShape( ylwBlock, -1, rgbaColor=[1.0, 1.0, 0.0, 1] )

        bluBlock = make_block()
        pb.changeVisualShape( bluBlock, -1, rgbaColor=[0.0, 0.0, 1.0, 1] )

        grnBlock = make_block()
        pb.changeVisualShape( grnBlock, -1, rgbaColor=[0.0, 1.0, 0.0, 1] )

        ornBlock = make_block()
        pb.changeVisualShape( ornBlock, -1, rgbaColor=[1.0, 0.5, 0.0, 1] )

        vioBlock = make_block()
        pb.changeVisualShape( vioBlock, -1, rgbaColor=[0.5, 0.0, 1.0, 1] )

        self.blocks = [redBlock, ylwBlock, bluBlock, grnBlock, ornBlock, vioBlock, None]

    def reset_blocks( self ):
        """ Send blocks to random locations """
        for blockHandl in self.blocks:
            if blockHandl is not None:
                posn, ornt = rand_table_pose()
                pb.resetBasePositionAndOrientation( blockHandl, posn, ornt )

    def get_handle( self, name ):
        """ Get the ID of the requested object by `name` """
        if name in _BLOCK_NAMES:
            return self.blocks[ _BLOCK_NAMES.index( name ) ]
        else:
            return None
        
    def get_handle_at_pose( self, rowVec, posnErr = _POSN_STDDEV*2.0 ):
        """ Return the handle of the object nearest to the `rowVec` pose if it is within `posnErr`, Otherwise return `None` """
        posnQ, _ = row_vec_to_pb_posn_ornt( rowVec )
        distMin = 1e6
        indxMin = -1
        for i, blk in enumerate( self.blocks ):
            if blk is not None:
                blockPos, _ = pb.getBasePositionAndOrientation( blk )
                dist = np.linalg.norm( np.array( posnQ ) - np.array( blockPos ) )
                if dist < distMin:
                    distMin = dist
                    indxMin = i
        if (indxMin > -1) and (distMin <= posnErr):
            return self.blocks[ indxMin ]
        return None

    def step( self ):
        """ Advance one step and sleep """
        pb.stepSimulation()
        time.sleep( self.tIncr )
        ePsn, _    = self.robot.get_current_pose()
        for obj in self.grasp:
            pb.resetBasePositionAndOrientation( obj[0], np.add( obj[1], ePsn ), obj[2] )

    def spin_for( self, N = 1000 ):
        """ Run for `N` steps """
        for _ in range(N):
            self.step()

    def stop( self ):
        """ Disconnect from the simulation """
        pb.disconnect()

    def get_block_true( self, blockName ):
        """ Find one of the ROYGBV blocks, Fully Observable, Return None if the name is not in the world """
        try:
            idx = _BLOCK_NAMES.index( blockName )
            blockPos, blockOrn = pb.getBasePositionAndOrientation( self.blocks[idx] )
            blockPos = np.array( blockPos )
            return ObjectSymbol( 
                DummyBelief( blockName ), 
                blockName, 
                pb_posn_ornt_to_row_vec( blockPos, blockOrn ) 
            )
        except ValueError:
            return None
        
    def full_scan_true( self ):
        """ Find all of the ROYGBV blocks, Fully Observable """
        rtnSym = []
        for name in _BLOCK_NAMES[:-1]:
            rtnSym.append( self.get_block_true( name ) )
        return rtnSym
        
    def get_block_grasp_true( self, blockName ):
        """ Find a grasp for one of the ROYGBV blocks, Fully Observable, Return None if the name is not in the world """
        symb = self.get_block_true( blockName )
        pose = symb.pose
        pose[2] += _GRASP_VERT_OFFSET
        posn, _ = row_vec_to_pb_posn_ornt( pose )
        ornt = _GRASP_ORNT_XYZW.copy()
        return pb_posn_ornt_to_row_vec( posn, ornt )
    
    def robot_grasp_block( self, blockName ):
        """ Lock the block to the end effector """
        hndl = self.get_handle( blockName )
        symb = self.get_block_true( blockName )
        bPsn, bOrn = row_vec_to_pb_posn_ornt( symb.pose )
        ePsn, _    = self.robot.get_current_pose()
        pDif = np.subtract( bPsn, ePsn )
        self.grasp.append( (hndl,pDif,bOrn,) ) # Preserve the original orientation because I am lazy

    def robot_release_all( self ):
        """ Unlock all objects from end effector """
        self.grasp = []

    def get_block_noisy( self, blockName, confuseProb = 0.10, poseStddev = _POSN_STDDEV ):
        """ Find one of the ROYGBV blocks, Partially Observable, Return None if the name is not in the world """
        try:
            idx = _BLOCK_NAMES.index( blockName )
            blockPos, blockOrn = pb.getBasePositionAndOrientation( self.blocks[idx] )
            blockPos = np.array( blockPos ) + np.array( [np.random.normal( 0.0, poseStddev/3.0 ) for _ in range(3)] )
            rtnObj = ObjectBelief()
            rtnObj.pose = pb_posn_ornt_to_row_vec( blockPos, blockOrn )
            for i in range( len( _BLOCK_NAMES ) ):
                blkName_i = _BLOCK_NAMES[i]
                if blkName_i == blockName:
                    rtnObj.labels[ blkName_i ] = 1.0-confuseProb*(len( _BLOCK_NAMES )-1)
                else:
                    rtnObj.labels[ blkName_i ] = confuseProb
            return rtnObj
        except ValueError:
            return None
        
    def full_scan_noisy( self, confuseProb = 0.10, poseStddev = _POSN_STDDEV ):
        """ Find all of the ROYGBV blocks, Partially Observable """
        rtnBel = []
        for name in _BLOCK_NAMES[:-1]:
            rtnBel.append( self.get_block_noisy( name, confuseProb, poseStddev ) )
        return rtnBel
    
    def get_handle_name( self, handle ):
        """ Get the block name that corresponds to the handle """
        try:
            idx = self.blocks.index( handle )
            return _BLOCK_NAMES[ idx ]
        except ValueError:
            return None

    def check_predicate( self, symbol, posnErr = _POSN_STDDEV*2.0 ):
        """ Check that the `symbol` is True """
        handle = self.get_handle_at_pose( symbol.pose, posnErr )
        return (self.get_handle_name( handle ) == symbol.label)
    
    def validate_goal_spec( self, spec, posnErr = _POSN_STDDEV*2.0 ):
        """ Return true only if all the predicates in `spec` are true """
        for p in spec:
            if not self.check_predicate( p, posnErr ):
                return False
        return True


########## UTILITY CLASSES & SYMBOLS ###############################################################
_POSN_STDDEV = 0.008
_NULL_NAME   = "NOTHING"
_NULL_THRESH = 0.75
_N_POSE_UPDT = 25


class ObjectSymbol:
    """ Determinized object """

    def __init__( self, ref, label, pose ):
        """ Assign members """
        self.ref    = ref # - Belief from which this symbols was sampled
        self.label  = label # Sampled object label
        self.pose   = pose #- Sampled object pose
        self.action = None #- Action to which this symbol was assigned

    def prob( self ):
        """ Get the current belief this symbol is true based on the belief this symbol was drawn from """
        return self.ref.labels[self.label]
    
    def __repr__( self ):
        """ String representation, Including current symbol belief """
        return f"<{self.label} @ {self.pose}, P={self.prob() if (self.ref is not None) else None}>"
    
    def p_attached( self ):
        """ Return true if this symbol has been assigned to an action """
        return (self.action is not None)
    

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
        if isinstance( poseOrBelief, (ObjectBelief, ObjectSymbol) ):
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
    
    def sample_symbol( self ):
        """ Sample a determinized symbol from the hybrid distribution """
        label = roll_outcome( self.labels )
        try:
            poseSample = np.random.multivariate_normal( self.pose, self.covar ) 
        except np.linalg.LinAlgError:
            self.reset_covar()
            poseSample = np.random.multivariate_normal( self.pose, self.covar ) 
        return ObjectSymbol( 
            self,
            label, 
            poseSample
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


########## MOCK PLANNER ############################################################################

##### Mock Action #########################################################
_PROB_TICK_FAIL = 0.01 # 0.20
_Z_SAFE         = 8.0*_BLOCK_SCALE

def MockAction( objName, dest, world = None, robot = None ):
    """ Create the Least Behavior """

    # posnTgt, orntTgt = row_vec_to_pb_posn_ornt( origin_row_vec() )
    
    bt = Sequence() # Memory by default
    
    setattr( bt, "objName", objName ) # Type of object required
    setattr( bt, "dest"   , dest    ) # Destination pose
    setattr( bt, "symbol" , None    ) # Symbol on which this behavior relies
    setattr( bt, "msg"    , ""      ) # Message: Reason this action failed -or- OTHER
    setattr( bt, "ctrl"   , robot   ) # Message: Reason this action failed -or- OTHER
    setattr( bt, "world"  , world   ) # Message: Reason this action failed -or- OTHER
    
    return bt

def prep_action( action, world, robot ):
    """ Set appropriate target """
    posnEnd, orntEnd = row_vec_to_pb_posn_ornt( action.dest )
    graspPose = action.symbol.pose
    handle = world.get_handle_at_pose( graspPose )

    miniGoal = ObjectSymbol( None, action.objName, action.dest )

    if world.check_predicate( miniGoal, _ACCEPT_POSN_ERR ):
        print( f"{action.name} ALREADY DONE" )
        action.status = Status.SUCCESS
        return

    if handle is not None:
        targetNam = world.get_handle_name( handle )

        posnTgt, orntTgt = row_vec_to_pb_posn_ornt( graspPose )
        
        posnTgt[2] += _GRASP_VERT_OFFSET
        orntTgt = _GRASP_ORNT_XYZW.copy()
        
        posnEnd[2] += _GRASP_VERT_OFFSET
        orntEnd = orntTgt[:]

        action.add_child( Pick_at_Pose( posnTgt, orntTgt, targetNam, zSAFE = _Z_SAFE, name = "Pick_at_Pose", 
                                    ctrl = robot, world = world ) )
        action.add_child( Place_at_Pose( posnEnd, orntEnd, zSAFE = _Z_SAFE, name = "Place_at_Pose", 
                                    ctrl = robot, world = world ) )
    else:
        action.status = Status.FAILURE
        action.msg    = "Object miss"
    
def prep_plan( plan, world, robot ):
    """ Set appropriate targets """
    for action in plan:
        prep_action( action, world, robot )


def copy_action( action ):
    """ Deep copy """
    rtnObj = MockAction( action.objName, action.dest, action.world, action.ctrl )
    rtnObj.status = action.status
    rtnObj.symbol = action.symbol
    return rtnObj


def get_grounded_action( action, symbol ):
    """ Copy action with a symbol attached """
    rtnAct = MockAction( action.objName, action.dest, action.world, action.ctrl )
    rtnAct.symbol = symbol
    symbol.action = rtnAct
    return rtnAct
    
def set_action_ground( action, symbol ):
    """ Attach symbol """
    action.symbol = symbol
    symbol.action = action
    
def p_grounded( action ):
    """ Return true if a symbol was assigned to this action """
    return (action.symbol is not None)

def repr_action( action ):
    """ Text representation """
    return f"[{action.objName} --to-> {action.dest}, Symbol: {action.symbol}]"
    
def cost( action ):
    """ Get the linear distance between the symbol pose and the destination """
    # print( self.dest, '\n', row_vec_to_homog( self.symbol.pose ) )
    return translation_diff( row_vec_to_homog( action.dest ), row_vec_to_homog( action.symbol.pose ) )
    


##### Planner Helpers #####################################################

def p_plan_grounded( plan ):
    """ Return true if every action in the plan is grounded, Otherwise return False """
    for action in plan:
        if not p_grounded( action ):
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
        total += cost( action )
    return total

def release_plan_symbols( plan ):
    """ Detach symbols from all actions in the plan """
    for action in plan:
        if action.symbol is not None:
            action.symbol.action = None
            action.symbol = None

def setup_plan_for_running( plan, world, robot ):
    """ Connect the plan to world and robot """
    connect_BT_to_robot_world( plan, robot, world )
    plan.setup_with_descendants()
    


##### Mock Planner ########################################################
_LOG_PROB_FACTOR = 10.0
_LOG_BASE        =  2.0
_PLAN_THRESH     =  0.02
_ACCEPT_POSN_ERR =  0.5*_BLOCK_SCALE 

class MockPlan( Sequence ):
    """ Special list with priority """

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
            rtnGoal.append( ObjectSymbol( None, action.objName, action.dest ) )
        return rtnGoal
    
    # # def tick( self, *args, **kwargs ):
    # def tick_once( self, *args, **kwargs ):
    #     for action in self.children:
    #         if action.status in (Status.INVALID, Status.RUNNING):
    #             miniGoal = ObjectSymbol( None, action.objName, action.dest )
    #             if self.world.check_predicate( miniGoal, _ACCEPT_POSN_ERR ):
    #                 print( f"{action.name} ALREADY DONE" )
    #                 action.status = Status.SUCCESS
    #     # super().tick_once( *args, **kwargs )
    #     # self.world.spin_for( 10 )
    #     # super().tick( )
    #     super().tick_once( *args, **kwargs )

##### METRICS #############################################################

g_BGN   = None
g_RUN   = False
metrics = {
    "N"     : 0,
    "pass"  : 0,
    "fail"  : 0,
    "trials": [],
}

def begin_trial():
    """ Increment number of trials and set state """
    global g_BGN, g_RUN, metrics
    g_BGN = now()
    g_RUN = True
    metrics['N'] += 1

def end_trial( p_pass, infoDict = None ):
    """ Record makespan and trial info """
    global g_BGN, g_RUN, metrics
    if infoDict is None:
        infoDict = {}
    runDct = {
        "makespan" : now() - g_BGN,
        "result"   : p_pass,
    }
    runDct.update( infoDict )
    metrics['trials'].append( runDct )
    if p_pass:
        metrics['pass'] += 1
    else:
        metrics['fail'] += 1

########## PLANNER #################################################################################

class MockPlanner:
    """ Least structure needed to compare plans """

    def reset_beliefs( self ):
        """ Erase belief memory """
        self.beliefs = [] # Distributions over objects
        self.symbols = []
        self.plans   = [] # PriorityQueue()

    def __init__( self, world ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.world   = world
        self.reset_beliefs()
        self.poses   = { # -- Intended destinations
            "P1" : [ _MIN_X_OFFSET+2.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ],
            "P2" : [ _MIN_X_OFFSET+4.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ],
            "P3" : [ _MIN_X_OFFSET+3.0*_BLOCK_SCALE, 0.000, 2.0*_BLOCK_SCALE,  1,0,0,0 ],
            "P4" : [ _MIN_X_OFFSET+6.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ],
            "P5" : [ _MIN_X_OFFSET+8.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ],
            "P6" : [ _MIN_X_OFFSET+7.0*_BLOCK_SCALE, 0.000, 2.0*_BLOCK_SCALE,  1,0,0,0 ],
        }
        skel1 = MockPlan()
        skel1.add_children( [MockAction('redBlock',self.poses['P1']),MockAction('ylwBlock',self.poses['P2']),MockAction('bluBlock',self.poses['P3']),] )
        skel2 = MockPlan()
        skel2.add_children( [MockAction('grnBlock',self.poses['P4']),MockAction('ornBlock',self.poses['P5']),MockAction('vioBlock',self.poses['P6']),] )
        self.skltns = [ # Plan skeletons, Each builds an arch
            skel1,
            skel2,
        ]

    def get_skeleton( self, idx ):
        """ Get the plan skeleton at `idx` """
        if idx < len( self.skltns ):
            rtnSkel = MockPlan()
            rtnSkel.goal = idx
            for action in self.skltns[ idx ]:
                rtnSkel.append( copy_action( action ) )
            return rtnSkel
        else:
            return list()

    def ground_plans_true( self ):
        """ Assign fully observable symbols to the plan skeletons """
        self.symbols = self.world.full_scan_true()
        for skeleton in self.skltns:
            plan = []
            for absAct in skeleton:
                for sym in self.symbols:
                    if sym.label == absAct.objName:
                        plan.append( absAct.get_grounded( sym ) )
                        break
            self.plans.append( plan )
        print( f"Formed {len(self.plans)} with {[len(pln) for pln in self.plans]} actions each!" )

    def exec_plans_true( self ):
        """ Execute fully observable plans """
        for plan in self.plans:
            for action in plan:
                print( f"Execute: {action}" )
                posn, ornt = row_vec_to_pb_posn_ornt( action.dest )
                pb.resetBasePositionAndOrientation( self.world.get_handle( action.objName ), posn, ornt )
                for _ in range(10):
                    self.world.step()

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
        if not relevant:
            self.beliefs.append( objBelief )
        return relevant
    
    def unvisit_beliefs( self ):
        """ Set visited flag to False for all beliefs """
        for belief in self.beliefs:
            belief.visited = False

    def check_symbol( self, symbol ):
        """ Return true if there is a belief that supports this symbol, Otherwise return false """
        for belief in self.beliefs:
            if belief.p_pose_relevant( symbol ):
                return True
        return False

    def exec_plans_noisy( self, N = 1200,  Npause = 500 ):
        """ Execute partially observable plans """

        self.world.reset_blocks()
        self.world.robot.goto_home()
        self.world.spin_for( Npause )

        self.reset_beliefs()

         # Number of iterations for this test
        K =    5 # Number of top plans to maintain
        ### Main Planner Loop ###  
        currPlan     = None
        achieved     = []
        trialMetrics = Counter()
        pPass        = False
        pBork        = False
        begin_trial()
        # 2023-12-11: For now, loop a limited number of times
        for i in range(N):

            ## Gather Evidence ##
            # 2023-12-11: For now, imagine a camera that always sees all the blocks


            truSym = self.world.full_scan_true()
            for truth in truSym:
                posn, _ = row_vec_to_pb_posn_ornt( truth.pose )
                for coord in posn:
                    if abs( coord ) >= 10.0:
                        trialMetrics[ "Simulation Fault" ] += 1
                        pBork = True

            if pBork:
                break

            objEvidence = self.world.full_scan_noisy()
            print( f"{i+1}: Got {len(objEvidence)} beliefs!" )

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
            
            ## Retain only fresh beliefs ##
            belObj = []
            for belief in self.beliefs:
                if belief.visited:
                    belObj.append( belief )
                else:
                    belief.integrate_belief( belief.sample_nothing() )
                    if belief.labels[ _NULL_NAME ] < _NULL_THRESH:
                        belObj.append( belief )
                    else:
                        print( "Belief DESTROYED!" )
            self.beliefs = belObj

            ## Sample Symbols ##
            nuSym = [bel.sample_symbol() for bel in self.beliefs]

            ## Ground Plans ##
            svSym     = [] # Only retain symbols that were assigned to plans!
            skeletons = [self.get_skeleton( j ) for j in range( len( self.skltns ) )]
            for sym in nuSym:
                assigned = False
                for l, skel in enumerate( skeletons ):
                    for j, action in enumerate( skel ):
                        if not p_grounded( action ):
                            if (action.objName == sym.label) and (not sym.p_attached()):
                                set_action_ground( action, sym )
                                assigned = True
                        if assigned:
                            break
                    if assigned:
                        break
                if sym.p_attached():
                    svSym.append( sym )
                
            for k, skel in enumerate( skeletons ):
                if p_plan_grounded( skel ):
                    self.plans.append( skel )
            self.symbols.extend( svSym )
            print( f"There are {len(self.plans)} plans!" )

            ## Grade Plans ##
            savPln = []
            for m, plan in enumerate( self.plans ):
                cost  = plan_cost( plan )
                prob  = plan_confidence( plan )
                score = cost - _LOG_PROB_FACTOR * log( prob, _LOG_BASE )
                plan.rank = score
                # Destroy (Degraded Plans || Plans with NaN Priority) #
                if (prob > _PLAN_THRESH) and (not isnan( score )):
                    savPln.append( plan )
                else:
                    release_plan_symbols( plan )
                    print( f"\tReleased {len(plan)} symbols!" )

            ## Enqueue Plans ##    
            savPln.sort()
            self.plans = savPln[:K]
            for badPlan in savPln[K:]:
                release_plan_symbols( badPlan )
            for m, plan in enumerate( self.plans ):
                print( f"\tPlan {m+1} --> Cost: {cost}, P = {prob}, {'Retain' if (prob > _PLAN_THRESH) else 'DELETE'}, Priority = {plan.rank}" )

            ## Destroy Unlikely Symbols ##
            savSym = [] # Only save likely symbols attached to plans
            cDel   = 0
            for sym in self.symbols:
                if (sym.prob() > _PLAN_THRESH) and sym.p_attached():
                    savSym.append( sym )
                else:
                    cDel += 1
            self.symbols = savSym
            print( f"Retained {len(self.symbols)} symbols, and deleted {cDel}!" )

            ## Execute Current Plan ##
            # Pop top plan
            if (currPlan is None) and len( self.plans ):
                try:

                    currPlan = self.plans[0]
                    prep_plan( currPlan, self.world, self.world.robot )
                    setup_plan_for_running( currPlan, self.world, self.world.robot )
                    self.plans.pop(0)

                    while currPlan.goal in achieved:
                        if currPlan is not None:
                            release_plan_symbols( currPlan )

                        currPlan = self.plans[0]
                        prep_plan( currPlan, self.world, self.world.robot )
                        setup_plan_for_running( currPlan, self.world, self.world.robot )
                        self.plans.pop(0)

                except (IndexError, AttributeError):
                    if currPlan is not None:
                        release_plan_symbols( currPlan )
                    currPlan = None
            if currPlan is not None:
                if currPlan.status == Status.SUCCESS:
                    achieved.append( currPlan.goal )
                    release_plan_symbols( currPlan )
                    currPlan = None

                elif currPlan.status == Status.FAILURE:
                    print( f"TRASHING failed plan: {currPlan}" )
                    trialMetrics[ currPlan.msg ] += 1
                    release_plan_symbols( currPlan )
                    world.robot_release_all()
                    currPlan = None

                elif plan_confidence( currPlan ) >= _PLAN_THRESH:
                    # currPlan.tick( self.world, _ACCEPT_POSN_ERR )
                    # print( currPlan.ctrl, currPlan.world )
                    
                    ## Step ##
                    self.world.spin_for( 10 )
                    currPlan.tick_once()
                    # currPlan.tick()

                    if random() < _PROB_TICK_FAIL:
                        currPlan.status = Status.FAILURE
                        currPlan.msg    = "Action Fault"

                else:
                    print( f"TRASHING unlikely plan: {currPlan}" )
                    trialMetrics[ "Unlikely Symbol" ] += 1
                    release_plan_symbols( currPlan )
                    currPlan = None

            ## Check Win Condition ##
            nuChieved = []
            for goalNum in achieved:
                skeleton = self.skltns[ goalNum ]
                goal     = skeleton.get_goal_spec()
                solved   = self.world.validate_goal_spec( goal, _ACCEPT_POSN_ERR )
                if solved:
                    print( f"Goal {goalNum} is SOLVED!" )
                    nuChieved.append( goalNum )
                else:
                    trialMetrics[ "Goal NOT Met" ] += 1
            achieved = nuChieved

            if len( achieved ) >= len( self.skltns ):
                break

            
            print()
            
        pPass = (len( achieved ) >= len( self.skltns ))

        if pPass:
            print( "\n### GOALS MET ###\n" )
        elif pBork:
            print( "\n!!! SIM BORKED !!!\n" )
        else:
            print( "\n### TIMEOUT ###\n" )

        for k, v in trialMetrics.items():
            print( f"Failure: {k}, Occurrences: {v}" )

        end_trial( pPass, trialMetrics )
        

########## MAIN ####################################################################################

##### Env. Settings #####
np.set_printoptions( precision = 3, linewidth = 145 )


##### Run Sim #####
if __name__ == "__main__":

    ### Init ###

    world   = PB_BlocksWorld()
    robot = world.robot

    planner = MockPlanner( world )
    Nruns   = 250
    
    ### Trials ###
    for i in range( Nruns ):
        print(f'\n##### Trial {i+1} of {Nruns} #####')
        planner.exec_plans_noisy( 1200 )
        world.spin_for( 200 )
        print('\n')

    ### Analyze ###
    import matplotlib.pyplot as plt

    print( f"Success Rate __ : {metrics['pass']/metrics['N']}" )
    spans = [ dct['makespan'] for dct in metrics["trials"] ]
    avgMs = sum( spans ) / metrics['N']
    print( f"Average Makespan: {avgMs}" )

    Nbins = 10

    msPass = []
    msFail = []

    for trial in metrics["trials"]:
        if trial['result']:
            msPass.append( trial['makespan'] )
        else:
            msFail.append( trial['makespan'] )

    with open( 'robotDemo250_2024-01-31.pkl', 'wb' ) as handle:
        pickle.dump( metrics, handle )

    with open( 'robotDemo250_2024-01-31_msPass.pkl', 'wb' ) as handle:
        pickle.dump( msPass, handle )       

    with open( 'robotDemo250_2024-01-31_msFail.pkl', 'wb' ) as handle:
        pickle.dump( msFail, handle )    

    plt.hist( [msPass, msFail], Nbins, histtype='bar', label=["Success", "Failure"] )

    plt.legend(); plt.xlabel('Episode Makespan'); plt.ylabel('Count')
    plt.savefig( 'robotDemo_Makespan.pdf' )

    plt.show()
