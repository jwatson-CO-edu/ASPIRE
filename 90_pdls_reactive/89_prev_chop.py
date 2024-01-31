########## INIT ####################################################################################

##### Imports #####
import sys, pickle
from collections import Counter
from math import log, isnan

sys.path.append( "../" )
from magpie.poses import translation_diff

from utils import *
from pb_BT import *





        







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

    def __init__( self, world ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.world   = world
        self.beliefs = [] # Distributions over objects
        self.symbols = []
        self.plans   = [] # PriorityQueue()
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

    def exec_plans_noisy( self, N = 1200,  Npause = 200 ):
        """ Execute partially observable plans """

        self.world.reset_blocks()
        self.world.spin_for( Npause )

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
    robot.goto_home()
    world.spin_for( 500 )

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
