""" ##### DEV PLAN #####
[ ] Measure failure rate
[ ] Measure makespan
"""

########## INIT ####################################################################################

import time, sys, pickle
now = time.time

from random import random, choice
from math import log
from collections import Counter
from pprint import pprint

import numpy as np

import pybullet as pb
import pybullet_data

from spatialmath.base import r2q
from spatialmath.quaternion import UnitQuaternion

from scipy.stats import chi2

sys.path.append( "../" )
from magpie.poses import translation_diff


########## UTILITY FUNCTIONS #######################################################################

def pb_posn_ornt_to_homog( posn, ornt ):
    """ Express the PyBullet position and orientation as homogeneous coords """
    H = np.eye(4)
    Q = UnitQuaternion( ornt[-1], ornt[:3] )
    H[0:3,0:3] = Q.SO3().R
    H[0:3,3]   = np.array( posn )
    return H


def pb_posn_ornt_to_row_vec( posn, ornt ):
    """ Express the PyBullet position and orientation as a Position and Orientation --> [Px,Py,Pz,Ow,Ox,Oy,Oz] """
    V = list( posn[:] )
    V.append( ornt[-1] )
    V.extend( ornt[:3] )
    return np.array(V)


def row_vec_to_pb_posn_ornt( V ):
    """ Express the PyBullet position and orientation as a Position and Orientation --> [Px,Py,Pz,Ow,Ox,Oy,Oz] """
    posn = np.array( V[0:3] )
    ornt = np.zeros( (4,) )
    ornt[:3] = V[4:7]
    ornt[-1] = V[3]
    return posn, ornt


def row_vec_to_homog( V ):
    """ Express [Px,Py,Pz,Ow,Ox,Oy,Oz] as homogeneous coordinates """
    posn, ornt = row_vec_to_pb_posn_ornt( V )
    return pb_posn_ornt_to_homog( posn, ornt )


def homog_to_row_vec( homog ):
    """ Express a homogeneous coord as a Position and Orientation --> [Px,Py,Pz,Ow,Ox,Oy,Oz] """
    P = homog[0:3,3]
    Q = UnitQuaternion( r2q( homog[0:3,0:3] ) )
    V = np.zeros( (7,) )
    V[0:3] = P[:]
    V[3]   = Q.s
    V[4:7] = Q.v[:]
    return np.array(V)


def total_pop( odds ):
    """ Sum over all categories in the prior odds """
    total = 0
    for k in odds:
        total += odds[k]
    return total


def normalize_dist( odds_ ):
    """ Normalize the distribution so that the sum equals 1.0 """
    total  = total_pop( odds_ )
    rtnDst = dict()
    for k in odds_:
        rtnDst[k] = odds_[k] / total
    return rtnDst


def roll_outcome( odds ):
    """ Get a random outcome from the distribution """
    oddsNorm = normalize_dist( odds )
    distrib  = []
    outcome  = []
    total    = 0.0
    for o, p in oddsNorm.items():
        total += p
        distrib.append( total )
        outcome.append( o )
    roll = random()
    for i, p in enumerate( distrib ):
        if roll <= p:
            return outcome[i]
    return None


def get_confusion_matx( Nclass, confuseProb = 0.10 ):
    """ Get the confusion matrix from the label list """
    Pt = 1.0-confuseProb*(Nclass-1)
    Pf = confuseProb
    rtnMtx = np.eye( Nclass )
    for i in range( Nclass ):
        for j in range( Nclass ):
            if i == j:
                rtnMtx[i,j] = Pt
            else:
                rtnMtx[i,j] = Pf
    return rtnMtx


def multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc ):
    """ Update the prior belief using probabilistic evidence given the weight of that evidence """
    Nclass = cnfMtx.shape[0]
    priorB = np.array( priorB ).reshape( (Nclass,1,) )
    evidnc = np.array( evidnc ).reshape( (Nclass,1,) )
    P_e    = cnfMtx.dot( priorB ).reshape( (Nclass,) )
    P_hGe  = np.zeros( (Nclass,Nclass,) )
    for i in range( Nclass ):
        P_hGe[i,:] = (cnfMtx[i,:]*priorB[i,0]).reshape( (Nclass,) ) / P_e
    return P_hGe.dot( evidnc ).reshape( (Nclass,) )



########## UTILITY CLASSES & SYMBOLS ###############################################################
_POSN_STDDEV = 0.008
_NULL_NAME   = "NOTHING"
_NULL_THRESH = 0.75
_N_POSE_UPDT = 25


class SimpleBlock:
    """ Use this to ground the RYB blocks """
    def __init__( self, name, pcd, pose ):
        self.name = name
        self.pcd  = pcd
        self.pose = pose


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





########## ENVIRONMENT #############################################################################
_BLOCK_NAMES  = ['redBlock', 'ylwBlock', 'bluBlock', 'grnBlock', 'ornBlock', 'vioBlock', _NULL_NAME]

class DummyBelief:
    """ Stand-in for an actual `ObjectBelief` """
    def __init__( self, label ):
        self.labels = { label: 1.0 }

class PB_BlocksWorld:
    """ Simple physics simulation with 3 blocks """

    def __init__( self ):
        """ Create objects """
        self.physicsClient = pb.connect( pb.GUI ) # or pb.DIRECT for non-graphical version
        pb.setAdditionalSearchPath( pybullet_data.getDataPath() ) #optionally
        pb.setGravity( 0, 0, -10 )
        self.planeId = pb.loadURDF( "plane.urdf" )

        redBlock = pb.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        pb.changeVisualShape( redBlock, -1, rgbaColor=[1.0, 0.0, 0.0, 1] )

        ylwBlock = pb.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        pb.changeVisualShape( ylwBlock, -1, rgbaColor=[1.0, 1.0, 0.0, 1] )

        bluBlock = pb.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        pb.changeVisualShape( bluBlock, -1, rgbaColor=[0.0, 0.0, 1.0, 1] )

        grnBlock = pb.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        pb.changeVisualShape( grnBlock, -1, rgbaColor=[0.0, 1.0, 0.0, 1] )

        ornBlock = pb.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        pb.changeVisualShape( ornBlock, -1, rgbaColor=[1.0, 0.5, 0.0, 1] )

        vioBlock = pb.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
        pb.changeVisualShape( vioBlock, -1, rgbaColor=[0.5, 0.0, 1.0, 1] )

        self.blocks = [redBlock, ylwBlock, bluBlock, grnBlock, ornBlock, vioBlock, None]

        for _ in range( 100 ):
            pb.stepSimulation()
        print('\n')

    def reset_blocks( self ):
        """ Send blocks to random locations """
        for blockHandl in self.blocks:
            if blockHandl is not None:
                posn, ornt = row_vec_to_pb_posn_ornt( [ random()*3.0-1.5, random()*3.0-1.5, 0.150,1,0,0,0] )
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
        time.sleep( 1.0 / 240.0 )

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



########## MOCK PLANNER ############################################################################

##### Mock Action #########################################################
_PROB_TICK_FAIL = 0.01

class MockAction:
    """ Least Behavior """

    def __init__( self, objName, dest ):
        """ Init action without grounding """
        self.objName = objName # - Type of object required
        self.handle  = None
        self.dest    = dest # ---- Where we will place this object
        self.status  = "INVALID" # Current status of this behavior
        self.symbol  = None # ---- Symbol on which this behavior relies
        self.msg     = "" # ------ Message: Reason this action failed -or- OTHER


    def set_wp( self ):
        """ Build waypoints for the action """
        self.tStep  =  0
        self.tDex   =  0
        self.tDiv   = 10
        self.waypnt = []
        liftDist     = 0.500
        if self.p_grounded():    
            self.waypnt.append( self.symbol.pose )
            p1 = self.symbol.pose.copy()
            p1[2] += liftDist
            self.waypnt.append( p1 )
            p2 = self.dest.copy()
            p2[2] += liftDist
            self.waypnt.append( p2 )
            self.waypnt.append( self.dest.copy() )

    def copy( self ):
        """ Deep copy """
        rtnObj = MockAction( self.objName, self.dest )
        rtnObj.status = self.status
        rtnObj.symbol = self.symbol
        return rtnObj

    def get_grounded( self, symbol ):
        """ Copy action with a symbol attached """
        rtnAct = MockAction( self.objName, self.dest[:] )
        rtnAct.symbol = symbol
        rtnAct.set_wp()
        symbol.action = rtnAct
        return rtnAct
    
    def set_ground( self, symbol ):
        """ Attach symbol """
        self.symbol = symbol
        symbol.action = self
        self.set_wp()
    
    def p_grounded( self ):
        """ Return true if a symbol was assigned to this action """
        return (self.symbol is not None)

    def __repr__( self ):
        """ Text representation """
        return f"[{self.objName} --to-> {self.dest}, Symbol: {self.symbol}]"
    
    def cost( self ):
        """ Get the linear distance between the symbol pose and the destination """
        # print( self.dest, '\n', row_vec_to_homog( self.symbol.pose ) )
        return translation_diff( row_vec_to_homog( self.dest ), row_vec_to_homog( self.symbol.pose ) )
    
    def tick( self, world ):
        """ Animate an action """
        print( f"\t\tTick: {self.status}, {self}" )
        if random() < _PROB_TICK_FAIL:
            self.status = "FAILURE"
            self.msg    = "Action Fault"
        if self.status == "INVALID":
            self.status = "RUNNING"
            if self.p_grounded():
                # print( type( self.symbol.pose ) )
                self.handle = world.get_handle_at_pose( self.symbol.pose )
        if self.status == "RUNNING":
            if self.handle is None:
                self.status = "FAILURE"
                self.msg    = "Object miss"
            elif self.p_grounded():
                posn, ornt = row_vec_to_pb_posn_ornt( self.waypnt[self.tDex] )
                pb.resetBasePositionAndOrientation( self.handle, posn, ornt )
                self.tStep += 1
                if (self.tStep % self.tDiv == 0):
                    self.tDex += 1
                if self.tDex >= len( self.waypnt ):
                    self.status = "COMPLETE"
            else:
                self.status = "FAILURE"
                self.msg    = "No symbol"


##### Planner Helpers #####################################################

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


##### Mock Planner ########################################################
_LOG_PROB_FACTOR = 10.0
_LOG_BASE        =  2.0
_PLAN_THRESH     =  0.02
_ACCEPT_POSN_ERR =  0.075 # 0.150

class MockPlan( list ):
    """ Special list with priority """

    def __init__( self, *args, **kwargs ):
        """ Set default priority """
        super().__init__( *args, **kwargs )
        self.rank   = 0.0 # -------------- Priority of this plan
        self.rand   = random() * 10000.0 # Tie-breaker for sorting
        self.goal   = -1 # --------------- Goal that this plan satisfies if completed
        self.status = "INVALID" # -------- Current status of this plan
        self.idx    = -1 # --------------- Current index of the running action
        self.msg    = "" # --------------- Message: Reason this plan failed -or- OTHER

    def __lt__( self, other ):
        """ Compare to another plan """
        # Original Author: Jiew Meng, https://stackoverflow.com/a/9345618
        selfPriority  = (self.rank , self.rand )
        otherPriority = (other.rank, other.rand)
        return selfPriority < otherPriority
    
    def __repr__( self ):
        """ String representation of the plan """
        return f"<MockPlan, Goal: {self.goal}, Status: {self.status}, Index: {self.idx}>"
    
    def tick( self, world, posnErr = _POSN_STDDEV*2.0 ):
        """ Animate a plan """
        print( f"\tTick: {self}" )
        if self.status == "INVALID":
            self.status = "RUNNING"
            self.idx    = 0
        if self.status == "RUNNING":
            miniGoal = ObjectSymbol( None, self[ self.idx ].objName, self[ self.idx ].dest.copy() )
            if world.check_predicate( miniGoal, posnErr ):
                self[ self.idx ].status = "COMPLETE"
            else:
                self[ self.idx ].tick( world )
            cStat = self[ self.idx ].status
            cMssg = self[ self.idx ].msg
            if cStat == "COMPLETE":
                self.idx += 1
                if self.idx >= len( self ):
                    self.status = "COMPLETE"
            else:
                self.status = cStat
                if cStat == "FAILURE":
                    self.msg = f"(Action) {cMssg}"

    def get_goal_spec( self ):
        """ Get a fully specified goal for this plan """
        rtnGoal = []
        for action in self:
            rtnGoal.append( ObjectSymbol( None, action.objName, action.dest.copy() ) )
        return rtnGoal


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

    def __init__( self, world ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.world   = world
        self.beliefs = [] # Distributions over objects
        self.symbols = []
        self.plans   = [] # PriorityQueue()
        self.poses   = { # -- Intended destinations
            "P1" : [ 0.300,0.000,0.150,1,0,0,0],
            "P2" : [ 0.600,0.000,0.150,1,0,0,0],
            "P3" : [ 0.450,0.000,0.300,1,0,0,0],
            "P4" : [-0.300,0.000,0.150,1,0,0,0],
            "P5" : [-0.600,0.000,0.150,1,0,0,0],
            "P6" : [-0.450,0.000,0.300,1,0,0,0],
        }
        self.skltns = [ # Plan skeletons, Each builds an arch
            MockPlan([MockAction('redBlock',self.poses['P1']),MockAction('ylwBlock',self.poses['P2']),MockAction('bluBlock',self.poses['P3']),]),
            MockPlan([MockAction('grnBlock',self.poses['P4']),MockAction('ornBlock',self.poses['P5']),MockAction('vioBlock',self.poses['P6']),]),
        ]

    def get_skeleton( self, idx ):
        """ Get the plan skeleton at `idx` """
        if idx < len( self.skltns ):
            rtnSkel = MockPlan()
            rtnSkel.goal = idx
            for action in self.skltns[ idx ]:
                rtnSkel.append( action.copy() )
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

    def exec_plans_noisy( self, Npause = 200 ):
        """ Execute partially observable plans """

        self.world.reset_blocks()
        self.world.spin_for( Npause )

        N = 1200 # Number of iterations for this test
        K =    5 # Number of top plans to maintain
        ### Main Planner Loop ###  
        currPlan     = None
        achieved     = []
        trialMetrics = Counter()
        pPass        = False
        begin_trial()
        # 2023-12-11: For now, loop a limited number of times
        for i in range(N):

            ## Gather Evidence ##
            # 2023-12-11: For now, imagine a camera that always sees all the blocks
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
                        if not action.p_grounded():
                            if (action.objName == sym.label) and (not sym.p_attached()):
                                action.set_ground( sym )
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
                # Destroy Degraded Plans #
                if prob > _PLAN_THRESH:
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
                    self.plans.pop(0)
                    while currPlan.goal in achieved:
                        if currPlan is not None:
                            release_plan_symbols( currPlan )
                        currPlan = self.plans[0]
                        self.plans.pop(0)
                except (IndexError, AttributeError):
                    if currPlan is not None:
                        release_plan_symbols( currPlan )
                    currPlan = None
            if currPlan is not None:
                if currPlan.status == "COMPLETE":
                    achieved.append( currPlan.goal )
                    release_plan_symbols( currPlan )
                    currPlan = None
                elif currPlan.status == "FAILURE":
                    print( f"TRASHING failed plan: {currPlan}" )
                    trialMetrics[ currPlan.msg ] += 1
                    release_plan_symbols( currPlan )
                    currPlan = None
                elif plan_confidence( currPlan ) >= _PLAN_THRESH:
                    currPlan.tick( self.world, _ACCEPT_POSN_ERR )
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
                solved   = world.validate_goal_spec( goal, _ACCEPT_POSN_ERR )
                if solved:
                    print( f"Goal {goalNum} is SOLVED!" )
                    nuChieved.append( goalNum )
                else:
                    trialMetrics[ "Goal NOT Met" ] += 1
            achieved = nuChieved

            if len( achieved ) >= len( self.skltns ):
                break

            ## Step ##
            self.world.spin_for( 10 )
            print()
            
        pPass = (len( achieved ) >= len( self.skltns ))

        if pPass:
            print( "\n### GOALS MET ###\n" )
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
    planner = MockPlanner( world )
    Nruns   = 250
    
    ### Trials ###
    for i in range( Nruns ):
        print(f'\n##### Trial {i+1} of {Nruns} #####')
        planner.exec_plans_noisy()
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

    with open( 'fullDemo250_2024-01-26.pkl', 'wb' ) as handle:
        pickle.dumps( metrics, handle )

    with open( 'fullDemo250_2024-01-26_msPass.pkl', 'wb' ) as handle:
        pickle.dumps( msPass, handle )       

    with open( 'fullDemo250_2024-01-26_msFail.pkl', 'wb' ) as handle:
        pickle.dumps( msFail, handle )    

    plt.hist( [msPass, msFail], Nbins, histtype='bar', label=["Success", "Failure"] )

    plt.legend(); plt.xlabel('Episode Makespan'); plt.ylabel('Count')
    plt.savefig( 'fullDemo_Makespan.pdf' )

    plt.show()

    
    # histData = {}
    # failSet  = set([])
    # for trial in metrics["trials"]:
    #     for k, v in trial.items():
    #         failSet.add(k)

    # for k in failSet:
    #     histData[k] = []

    # for trial in metrics["trials"]:
    #     for k in failSet:
    #         if k in trial:
    #             histData[k].append( trial[k] )
    #         else:
    #             histData[k].append(0)

    

    # ocrMin = 10000
    # ocrMax = 0
    # for nam, lst in histData.items():
    #     if nam not in ['result', 'makespan']:
    #         ocrMin = min( ocrMin, min( lst ) )
    #         ocrMax = max( ocrMax, max( lst ) )
    
    # Lbins = np.linspace( ocrMin, ocrMax, Nbins, True )
    
    # test_data = []
    # labels    = []
    # for nam, lst in histData.items():
    #     if nam not in ['result', 'makespan']:
    #         labels.append( nam )
    #         test_data.append( lst )

    # x_positions = np.array( Lbins )

    # number_of_groups = len( test_data )
    # fill_factor =  .8  # ratio of the groups width
    #                 # relatively to the available space between ticks
    # bar_width = np.diff(x_positions).min()/number_of_groups * fill_factor


    # # labels = ['red flowers', 'yellow flowers', 'blue flowers']

    # plt.hist( test_data, Nbins, histtype='bar', label=labels )

    # # for i, groupdata in enumerate(test_data): 
    # #     bar_positions = [x_pos - number_of_groups*bar_width/2 + (i + 0.5)*bar_width for x_pos in x_positions]
    # #     plt.bar(bar_positions, groupdata, bar_width,
    # #             align='center',
    # #             linewidth=1, edgecolor='k',
    # #             alpha=0.7,
    # #             label=labels[i])

    # plt.xticks(x_positions)
    # plt.legend(); plt.xlabel('Occurrences During Episode'); plt.ylabel('Count')
    # plt.savefig( 'fullDemo.pdf' )
    

    # pprint( histData )

    
