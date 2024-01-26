########## INIT ####################################################################################

##### Imports #####

from utils import *
from pb_BT import *


##### Paths #####

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
TABLE_URDF_PATH = os.path.join( pybullet_data.getDataPath(), "table/table.urdf" )



########## DEV PLAN ################################################################################
"""
[>] Pick and Place behavior
"""



########## UR5 ROBOT ###############################################################################
_BLOCK_SCALE       = 0.038
_GRASP_VERT_OFFSET = _BLOCK_SCALE * 2.0
_GRASP_ORNT_XYZW   = np.array( [0, 0.7070727, 0, 0.7071408,] )
_Q_HOME            = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]
_ROT_VEL_SMALL     = 0.2

class UR5Sim:
    """ Original Author: Josep Daniel, https://github.com/josepdaniel/ur5-bullet/tree/master/UR5 """

    def load_robot( self ):
        """ Load UR5 from description """
        flags = 0 #pb.URDF_USE_SELF_COLLISION
        robot = pb.loadURDF( ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags )
        return robot
  
    def __init__( self, camera_attached=False ):
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
        print( vel )
        for v in vel:
            if v > _ROT_VEL_SMALL:
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
            maxNumIterations=1000,
            residualThreshold=0.005
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

def make_table():
    """ Load a table """
    # table = pb.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
    return pb.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])

def rand_table_pose():
    """ Return a random pose in the direct viscinity if the robot """
    return [ 
        0.280 + random()*8.0*_BLOCK_SCALE, 
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
                posn, ornt = row_vec_to_pb_posn_ornt( 
                    [ 0.070 + random()*10.0*_BLOCK_SCALE, 
                      0.070 + random()*10.0*_BLOCK_SCALE, 
                      _BLOCK_SCALE ,
                      1,0,0,0] 
                )
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

    def __init__( self, initStddev = _POSN_STDDEV ):
        """ Initialize with origin poses and uniform, independent variance """
        # stdDev = [initStddev if (i<3) else 0.0 for i in range(7)]
        stdDev = [initStddev for i in range(7)]
        self.labels  = {} # ---------------------- Current belief in each class
        self.pose    = np.array([0,0,0,1,0,0,0]) # Mean pose
        self.pStdDev = np.array(stdDev) # -------- Pose variance
        self.pHist   = [] # ---------------------- Recent history of poses
        self.pThresh = 0.5 # --------------------- Minimum prob density at which a nearby pose is relevant
        self.covar   = np.zeros( (7,7,) ) # ------ Pose covariance matrix
        for i, stdDev in enumerate( self.pStdDev ):
            self.covar[i,i] = stdDev * stdDev
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
        return ObjectSymbol( 
            self,
            label, 
            np.random.multivariate_normal( self.pose, self.covar ) 
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
        print( self.covar )
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




########## MAIN ####################################################################################
##### Env. Settings #####
np.set_printoptions( precision = 3, linewidth = 145 )
_Z_SAFE = 10.0*_BLOCK_SCALE


if __name__ == "__main__":

    ## Init ##
    target = 'ylwBlock'
    world = PB_BlocksWorld()
    robot = world.robot
    robot.goto_home()
    world.spin_for( 500 )
    
    graspPose = world.get_block_grasp_true( target )
    posnTgt, orntTgt = row_vec_to_pb_posn_ornt( graspPose )
    posnEnd, _       = rand_table_pose()
    posnEnd[2] += _GRASP_VERT_OFFSET
    orntEnd = orntTgt[:]

    print( "Waypoints" )
    print( posnTgt, orntTgt )
    print( posnEnd, orntEnd )

    bt = Sequence()
    bt.add_child( Pick_at_Pose( posnTgt, orntTgt, target, zSAFE = _Z_SAFE, name = "Pick_at_Pose", 
                                ctrl = robot, world = world ) )
    bt.add_child( Place_at_Pose( posnEnd, orntEnd, zSAFE = _Z_SAFE, name = "Place_at_Pose", 
                                ctrl = robot, world = world ) )
    run_BT_until_done( bt, world = world )

    robot.goto_home()

    world.spin_for( 4000 )