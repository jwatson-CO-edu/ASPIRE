########## DEV PLAN ################################################################################
"""
##### Repair #####
[N] Begin with some alternate setdown poses populated, 2024-03-08: Did NOT result in better performance
[ ] NEEDED: A way to detect when the solver has gone braindead!
"""



########## INIT ####################################################################################

##### Imports #####

### Standard ###
import sys, time, math
now = time.time
from traceback import print_exc, format_exc
from pprint import pprint

### Special ###
import numpy as np
from py_trees.common import Status


### Local ###

## PDDLStream ##
sys.path.append( "../pddlstream/" )
from pddlstream.algorithms.meta import solve
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import read, INF, get_file_path
from pddlstream.language.constants import print_solution, PDDLProblem

## MAGPIE ##
sys.path.append( "../" )
# from magpie.poses import translation_diff

from utils import ( row_vec_to_pb_posn_ornt, pb_posn_ornt_to_row_vec, diff_norm, closest_dist_Q_to_segment_AB,
                    DataLogger, )

from env_config import ( _GRASP_VERT_OFFSET, _GRASP_ORNT_XYZW, _NULL_NAME, _ACTUAL_NAMES, _MIN_X_OFFSET,
                         _NULL_THRESH, _BLOCK_SCALE, _CLOSEST_TO_BASE, _ACCEPT_POSN_ERR, _MIN_SEP, _Z_SAFE,
                         _N_POSE_UPDT, _WP_NAME, _SAMPLE_DET )

from PB_BlocksWorld import PB_BlocksWorld, rand_table_pose
from symbols import Object, Path

from beliefs import ObjectMemory
from actions import Plan, display_PDLS_plan, BT_Runner, get_ith_BT_action_from_PDLS_plan, Place, Stack



########## HELPER FUNCTIONS ########################################################################


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

########## EXECUTIVE (THE METHOD) ##################################################################

class ResponsiveExecutive:
    """ Least structure needed to compare plans """

    ##### Init ############################################################

    def reset_beliefs( self ):
        """ Erase belief memory """
        self.memory  = ObjectMemory() # Distributions over objects
        self.last    = {}


    def reset_state( self ):
        """ Erase problem state """
        self.symbols  = list()
        self.status   = Status.INVALID
        self.facts    = list()
        self.goal     = tuple()
        self.task     = None
        self.currPlan = None
        self.action   = None


    def __init__( self, world = None ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.world = world if (world is not None) else PB_BlocksWorld()
        self.reset_beliefs()
        self.reset_state()
        self.logger = DataLogger()
        # DEATH MONITOR
        self.noSoln =  0
        self.nonLim = 10


    ##### Stream Helpers ##################################################

    def sample_determ( self, objName ):
        """ Return the deterministic pose of the block """
        nuSym = self.world.full_scan_true()
        for sym in nuSym:
            if sym.label not in self.last:
                self.last[ sym.label ] = sym
        if objName in self.last:
            return self.last[ objName ]
        else:
            return None
        

    def calc_grasp( self, objPose ):
        """ A function that returns grasps """
        grasp_pose = list( objPose )
        grasp_pose[2] += _GRASP_VERT_OFFSET
        posn, _ = row_vec_to_pb_posn_ornt( grasp_pose )
        ornt = _GRASP_ORNT_XYZW.copy()
        return pb_posn_ornt_to_row_vec( posn, ornt )
    
        
    def calc_ik( self, effPose ):
        """ Helper function for IK and Path planners """
        currPosn, _        = self.world.robot.get_current_pose()
        grspPosn, grspOrnt = row_vec_to_pb_posn_ornt( effPose )
        if diff_norm( currPosn, grspPosn ) < 2.0:
            return self.world.robot.calculate_ik_quat( grspPosn, grspOrnt )
        else:
            return None
        

    def object_from_label_pose( self, objcName, objcPose ):
        """ Load a Waypoint with relevant data """
        grspPose = self.calc_grasp( objcPose )
        grspCnfg = self.calc_ik( grspPose )
        rtnObj   = Object( objcName, objcPose )
        rtnObj.grasp  = grspPose
        rtnObj.config = grspCnfg
        return rtnObj


    def add_grasp_config_to_object( self, obj ):
        """ Load an existing Waypoint with relevant data """
        obj.grasp  = self.calc_grasp( obj.pose )
        obj.config = self.calc_ik( obj.grasp )
    

    ##### Stream Creators #################################################

    def get_object_stream( self ):
        """ Return a function that returns poses """

        def stream_func( *args ):
            """ A function that returns poses """

            print( f"\nEvaluate OBJECT stream with args: {args}\n" )

            objcName = args[0]

            ## Sample Symbols ##
            if _SAMPLE_DET:
                rtnObj = self.sample_determ( objcName )
            else:
                # rtnObj = self.memory.sample_consistent( objcName )
                rtnObj = self.memory.sample_consistent_fresh( objcName )
            if (rtnObj is not None) and (rtnObj.label != _NULL_NAME):
                self.add_grasp_config_to_object( rtnObj )
                print( f"OBJECT stream SUCCESS: {rtnObj}\n" )
                yield (rtnObj,)

        return stream_func
    

    def safe_motion_test( self, bgn, end ):
        """ Test if a line between two effector positions passes through the robot base """
        posn1  , _       = row_vec_to_pb_posn_ornt( bgn.grasp )
        posn2  , _       = row_vec_to_pb_posn_ornt( end.grasp )
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
    

    def path_segment_checker( self, bgn, end ):
        """ Helper function for the path planner stream """

        print( bgn, end )
        if not self.safe_motion_test( bgn, end ):
            return False

        label      = bgn.label
        posnBgn, _ = row_vec_to_pb_posn_ornt( bgn.pose )
        posnEnd, _ = row_vec_to_pb_posn_ornt( end.pose )

        if label == _WP_NAME:
            return True

        if diff_norm( posnBgn, posnEnd ) > 0.0: 

            ## Sample Symbols ##
            if _SAMPLE_DET:
                nuSym = self.world.full_scan_true()
            else:
                nuSym = self.memory.scan_consistent_fresh()
                # nuSym = self.memory.scan_consistent()
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

        def stream_func( *args, fluents=[] ):
            
            print( f"\nEvaluate PATH stream with args: {args}\n" )
            print( f"\nEvaluate PATH stream with fluents: {fluents}\n" )

            obj1, obj2 = args

            print( obj1, obj2 )
            if self.path_segment_checker( obj1, obj2 ):
                yield ( Path( [obj1, obj2],), )
            else:
                posnBgn, orntEnd = row_vec_to_pb_posn_ornt( obj1.pose )
                posnEnd, _       = row_vec_to_pb_posn_ornt( obj2.pose )
                posnMid    = np.add( posnBgn, posnEnd ) / 2.0
                posnMid[2] = _Z_SAFE
                orntMid    = list( orntEnd )
                objcPose = pb_posn_ornt_to_row_vec( posnMid, orntMid )
                mid      = self.object_from_label_pose( obj1.label, objcPose )
                if self.path_segment_checker( obj1, mid ) and self.path_segment_checker( mid, obj2 ):
                    yield ( Path( [obj1, mid, obj2,],), )
        return stream_func
    

    def get_carry_planner( self ):
        """ Return a function that checks if the path is free from obstruction """

        def stream_func( *args, fluents=[] ):
            
            print( f"\nEvaluate CARRY stream with args: {args}\n" )
            print( f"\nEvaluate CARRY stream with fluents: {fluents}\n" )

            label, obj1, obj2 = args

            posnBgn, orntEnd = row_vec_to_pb_posn_ornt( obj1.pose )
            posnEnd, _       = row_vec_to_pb_posn_ornt( obj2.pose )
            posnMid    = np.add( posnBgn, posnEnd ) / 2.0
            posnMid[2] = _Z_SAFE
            orntMid    = orntEnd[:]
            objcPose = pb_posn_ornt_to_row_vec( posnMid, orntMid )
            mid      = self.object_from_label_pose( obj1.label, objcPose )
            if self.path_segment_checker( obj1, mid ) and self.path_segment_checker( mid, obj2 ):
                yield ( Path( [obj1, mid, obj2,],), )
        return stream_func
    
    

    def get_stacker( self ):
        """ Return a function that computes Inverse Kinematics for a pose """

        def stream_func( *args, fluents=[] ):
            """ A function that computes a stacking pose across 2 poses """

            print( f"\nEvaluate STACK PLACE stream with args: {args}\n" )
            print( f"\nEvaluate STACK PLACE stream with fluents: {fluents}\n" )

            objDn1, objDn2 = args

            posnDn1, orntDn1 = row_vec_to_pb_posn_ornt( objDn1.pose )
            posnDn2, _       = row_vec_to_pb_posn_ornt( objDn2.pose )
            dist   = diff_norm( posnDn1, posnDn2 )
            sepMax = 2.1*_BLOCK_SCALE
            zMax   = max( posnDn1[2], posnDn2[2] )
            if (dist <= sepMax) and (zMax < 1.25*_BLOCK_SCALE) and (objDn1.label != objDn2.label):
                posnUp    = np.add( posnDn1, posnDn2 ) / 2.0
                posnUp[2] = 2.0*_BLOCK_SCALE
                orntUp    = orntDn1[:]
                objUp     = self.object_from_label_pose( _WP_NAME, pb_posn_ornt_to_row_vec( posnUp, orntUp ) )
                print( f"STACK PLACE stream SUCCESS: {objUp.pose}\n" )
                yield (objUp,)
            else:
                print( f"STACK PLACE stream FAILURE: {dist} > {sepMax}, {zMax} > {1.1*_BLOCK_SCALE}, {objDn1.label} == {objDn2.label} \n" )

        return stream_func
    

    def get_free_placement_test( self ):
        """ Return a function that checks if the pose is free from obstruction """

        def test_func( *args ):
            
            print( f"\nEvaluate PLACEMENT test with args: {args}\n" )

            label, obj = args
            posn , _    = row_vec_to_pb_posn_ornt( obj.pose )

            ## Sample Symbols ##
            if _SAMPLE_DET:
                nuSym = self.world.full_scan_true()
            else:
                # nuSym = self.memory.scan_consistent()
                nuSym = self.memory.scan_consistent_fresh()
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
            

    ##### Goal Validation #################################################

    def validate_predicate_true( self, pred ):
        """ Check if the predicate is true """
        pTyp = pred[0]
        if pTyp == 'HandEmpty':
            print( f"HandEmpty: {self.world.grasp}" )
            return (len( self.world.grasp ) == 0)
        elif pTyp == 'GraspObj':
            pLbl = pred[1]
            pPos = pred[2]
            tObj = self.world.get_block_true( pLbl )
            print( pred )
            print( "GraspObj:", pPos.pose[:3], tObj.pose[:3] )
            print( f"GraspObj: {diff_norm( pPos.pose[:3], tObj.pose[:3] )} <= {_ACCEPT_POSN_ERR}" )
            return (diff_norm( pPos.pose[:3], tObj.pose[:3] ) <= _ACCEPT_POSN_ERR)
        elif pTyp == 'Supported':
            lblUp = pred[1]
            lblDn = pred[2]
            objUp = self.world.get_block_true( lblUp )
            objDn = self.world.get_block_true( lblDn )
            xySep = diff_norm( objUp.pose[:2], objDn.pose[:2] )
            zSep  = objUp.pose[2] - objDn.pose[2] # Signed value
            print( f"Supported, X-Y Sep: {xySep} <= {2.0*_BLOCK_SCALE}, Z Sep: {zSep} >= {1.35*_BLOCK_SCALE}" )
            return ((xySep <= 2.0*_BLOCK_SCALE) and (zSep >= 1.35*_BLOCK_SCALE))
        else:
            print( f"UNSUPPORTED predicate check!: {pTyp}" )
            return False
    
    def validate_goal_true( self, goal ):
        """ Check if the goal is met """
        if goal[0] == 'and':
            for g in goal[1:]:
                if not self.validate_predicate_true( g ):
                    return False
            return True
        else:
            raise ValueError( f"Unexpected goal format!: {goal}" )
        

    def get_met_goal_predicates_true( self, goal ):
        """ Return a list of goal predicates that have already been met """
        rtnPred = []
        if goal[0] == 'and':
            for g in goal[1:]:
                if self.validate_predicate_true( g ):
                    rtnPred.append( g )
        else:
            print( f"Unexpected goal format!: {goal}" )
        return rtnPred


    ##### Goal Validation (Noisy) #########################################

    def validate_predicate_noisy( self, pred, objs ):
        """ Check if the predicate is true """

        def get_by_name( name, syms ):
            """ Get the list item with a label matching `name` """
            for sym in syms:
                if sym.label == name:
                    return sym
            return None

        pTyp = pred[0]
        if pTyp == 'HandEmpty':
            print( f"HandEmpty: {self.world.grasp}" )
            return (len( self.world.grasp ) == 0)
        elif pTyp == 'GraspObj':
            pLbl = pred[1]
            pPos = pred[2]
            # tObj = self.memory.sample_consistent( pLbl )
            tObj = get_by_name( pLbl, objs )
            try:
                return (diff_norm( pPos.pose[:3], tObj.pose[:3] ) <= _ACCEPT_POSN_ERR)
            except Exception:
                return False
        elif pTyp == 'Supported':
            lblUp = pred[1]
            lblDn = pred[2]
            # objUp = self.memory.sample_consistent( lblUp )
            objUp = get_by_name( lblUp, objs )
            # objDn = self.memory.sample_consistent( lblDn )
            objDn = get_by_name( lblDn, objs )
            try:
                xySep = diff_norm( objUp.pose[:2], objDn.pose[:2] )
                zSep  = objUp.pose[2] - objDn.pose[2] # Signed value
                print( f"Supported, X-Y Sep: {xySep} <= {2.0*_BLOCK_SCALE}, Z Sep: {zSep} >= {1.35*_BLOCK_SCALE}" )
                return ((xySep <= 2.0*_BLOCK_SCALE) and (zSep >= 1.35*_BLOCK_SCALE))
            except Exception:
                return False
        else:
            print( f"UNSUPPORTED predicate check!: {pTyp}" )
            return False

    def validate_goal_noisy( self, goal, objs ):
        """ Check if the goal is met """
        if goal[0] == 'and':
            for g in goal[1:]:
                if not self.validate_predicate_noisy( g, objs ):
                    return False
            return True
        else:
            raise ValueError( f"Unexpected goal format!: {goal}" )
        
    def get_met_goal_predicates_noisy( self, goal, objs ):
        """ Return a list of goal predicates that have already been met """
        rtnPred = []
        if goal[0] == 'and':
            for g in goal[1:]:
                if self.validate_predicate_noisy( g, objs ):
                    rtnPred.append( g )
        else:
            print( f"Unexpected goal format!: {goal}" )
        return rtnPred


    ##### TAMP Helpers ####################################################
        
    def check_OOB( self, thresh_m = 10.0 ):
        """ Return true if any of the simulated objects are out of bounds """
        truSym = self.world.full_scan_true()
        for truth in truSym:
            posn, _ = row_vec_to_pb_posn_ornt( truth.pose )
            for coord in posn:
                if abs( coord ) >= thresh_m:
                    return True
        return False
    
    def p_open_true( self, pose ):
        posn , _    = row_vec_to_pb_posn_ornt( pose )
        nuSym = self.world.full_scan_true()
        for sym in nuSym:
            symPosn, _ = row_vec_to_pb_posn_ornt( sym.pose )
            if diff_norm( posn, symPosn ) < ( _MIN_SEP ):
                return False
        return True
    

    def get_N_open_spots( self, N ):
        """ Find N spots on the table where we are free to place blocks """
        spots = []
        for _ in range(N):
            posn, ornt = rand_table_pose()
            pose = pb_posn_ornt_to_row_vec( posn, ornt )
            while not self.p_open_true( pose ):
                posn, ornt = rand_table_pose()
                pose = pb_posn_ornt_to_row_vec( posn, ornt )
            spots.append( pose )
        return spots

    
    def get_init_predicates( self ):
        """ Get the goal predicates and the predicates that are known before runtime """

        print( 'Robot:', self.world.robot.get_name() )

        start   = self.object_from_label_pose( _WP_NAME, pb_posn_ornt_to_row_vec( *self.world.robot.get_current_pose() ) )
        trgtRed = self.object_from_label_pose( 'redBlock', [ _MIN_X_OFFSET+2.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )
        trgtYlw = self.object_from_label_pose( 'ylwBlock', [ _MIN_X_OFFSET+4.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )
        trgtGrn = self.object_from_label_pose( 'grnBlock', [ _MIN_X_OFFSET+6.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )
        trgtOrn = self.object_from_label_pose( 'ornBlock', [ _MIN_X_OFFSET+8.0*_BLOCK_SCALE, 0.000, 1.0*_BLOCK_SCALE,  1,0,0,0 ] )

        self.facts = [
            ## Init Predicates ##
            ('Waypoint', start),
            ('AtObj', start),
            ('HandEmpty',),
            ## Goal Predicates ##
            ('Waypoint', trgtRed),
            ('Waypoint', trgtYlw),
            ('Waypoint', trgtGrn),
            ('Waypoint', trgtOrn),
        ] 

        for body in _ACTUAL_NAMES:
            self.facts.append( ('Graspable', body) )

        # ## Populate some Open Spots ##
        # spots = self.get_N_open_spots( 3 )
        # for spot in spots:
        #     self.facts.append( ('Waypoint', self.object_from_label_pose( _WP_NAME, spot ),) )

        print( f"### Initial Symbols ###" )
        for sym in self.facts:
            print( f"\t{sym}" )

        self.goal = ( 'and',
            ('HandEmpty',),
            
            ('GraspObj', 'redBlock', trgtRed),
            ('GraspObj', 'ylwBlock', trgtYlw),
            ('Supported', 'bluBlock', 'redBlock'),
            ('Supported', 'bluBlock', 'ylwBlock'),

            ('GraspObj', 'grnBlock', trgtGrn),
            ('GraspObj', 'ornBlock', trgtOrn),
            ('Supported', 'vioBlock', 'grnBlock'),
            ('Supported', 'vioBlock', 'ornBlock'),
        ) 

        print( f"### Goal ###" )
        pprint( self.goal )
    

    def check_goal_objects( self, goal, symbols ):
        """ Return True if the labels mentioned in the goals are a subset of the determinized symbols """
        goalSet = set([])
        symbSet = set( [sym.label for sym in symbols] )
        for g in goal:
            if isinstance( g, (tuple, list) ):
                prdName = g[0]
                if prdName == 'GraspObj':
                    goalSet.add( g[1] )
                elif prdName == 'Supported':
                    goalSet.add( g[1] )
                    goalSet.add( g[2] )
                else:
                    continue
        return (goalSet <= symbSet)


    def pddlstream_from_problem( self ):
        """ Set up a PDDLStream problem with the UR5 """

        domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'stream.pddl'))

        constant_map = {}
    
    
        stream_map = {
            ### Symbol Streams ###
            'sample-object':    from_gen_fn( self.get_object_stream() ), 
            'find-safe-motion': from_gen_fn( self.get_path_planner()  ),
            'find-safe-carry':  from_gen_fn( self.get_carry_planner() ),
            'find-stack-place': from_gen_fn( self.get_stacker()       ),
            ### Symbol Tests ###
            'test-free-placment': from_test( self.get_free_placement_test() ),
        }

        print( "About to create problem ... " )
    
        return PDDLProblem( domain_pddl, constant_map, stream_pddl, stream_map, self.facts, self.goal )
        

    def p_failed( self ):
        """ Has the system encountered a failure? """
        return (self.status == Status.FAILURE)
    

    ##### Task And Motion Planning ########################################


    def set_table( self ):
        """ Get ready for an experiment """
        self.world.robot.goto_home()
        self.world.reset_blocks()
        self.world.spin_for( 500 )


    def phase_1_Perceive( self, Nscans ):
        """ Take in evidence and form beliefs """

        for _ in range( Nscans ):
            self.memory.belief_update( self.world.full_scan_noisy() ) # We need at least an initial set of beliefs in order to plan

        objs = self.memory.scan_consistent()

        print( f"Starting Objects:" )
        for obj in objs:
            print( f"\t{obj}" )

        self.symbols = objs
        self.status  = Status.RUNNING
    

    def phase_2_Ground_Symbols( self ):
        """ Get the necessary initial state and check for goals already met """
    
        self.get_init_predicates()

        if self.validate_goal_noisy( self.goal, self.symbols ):
            self.status = Status.SUCCESS
        elif not self.check_goal_objects( self.goal, self.symbols ):
            self.logger.log_event( "Required objects missing", str( self.symbols ) )   
            self.status = Status.FAILURE
        else:
            self.facts.extend( self.get_met_goal_predicates_noisy( self.goal, self.symbols ) )


    def phase_3_Plan_Task( self ):
        """ Attempt to solve the symbolic problem """

        self.task = self.pddlstream_from_problem()

        self.logger.log_event( "Begin Solver" )

        try:
            # 'ff-eager-pref': 20s
            # 'ff-ehc' : No Sol'n
            # 'goal-lazy' : Very long
            # 'dijkstra' : Very long
            # 'max-astar' : Very long
            # 'lmcut-astar' : No Sol'n
            # 'ff-astar' : 40s
            # 'ff-ehc' : Very long
            # 'ff-wastar1' : 30s
            # 'ff-wastar2' : 15s
            # 'ff-wastar4' : 10-15s, Fails sometimes
            # 'ff-wastar5' : 10-15s, Fails sometimes
            # 'cea-wastar1' : Fails often
            # 'cea-wastar3' : 15-20s, Fails sometimes
            # 'cea-wastar5' : Very long
            # 'ff-wastar3' : 7-15s

            # planner = 'ff-eager-pref' #'ff-eager-pref' # 'add-random-lazy' # 'ff-eager-tiebreak' #'goal-lazy' #'ff-eager'
            # DEFAULT SOLVER
            solution = solve( 
                self.task, 
                algorithm = "adaptive", #"focused", #"binding", #"incremental", #"adaptive", 
                max_skeletons = 50,
                max_time      = 80.0,
                unit_costs   = True, 
                unit_efforts = True,
                effort_weight = 10.0, #200.0, #100.0, #50.0, #20.0, # 5.0, # 2.0 #10.0,
                success_cost = 40,
                initial_complexity = 1,
                complexity_step = 1,
                search_sample_ratio = 1/1000, #1/1500, #1/5, #1/1000, #1/750 # 1/1000, #1/2000 #500, #1/2, # 1/500, #1/200, #1/10, #2, # 25 #1/25
                reorder = False, # Setting to false bare impacts sol'n time
                # planner = planner
                # stream_info = stream_info,
            )
            # print( "Solver has completed!\n\n\n" )
            print_solution( solution )
        except Exception as ex:
            self.logger.log_event( "SOLVER FAULT", format_exc() )
            self.status = Status.FAILURE
            print_exc()
            solution = (None, None, None)
            self.noSoln += 1 # DEATH MONITOR

        plan, cost, evaluations = solution
        if (plan is not None) and len( plan ):
            display_PDLS_plan( plan )
            # print( type( plan ) )
            # print( type( cost ) )
            # print( type( evaluations ) )
            # exit()
            self.currPlan = plan
            self.action   = get_BT_plan_until_block_change( plan, self.world )
            self.noSoln = 0 # DEATH MONITOR
        else:
            self.noSoln += 1 # DEATH MONITOR
            self.logger.log_event( "NO SOLUTION" )
            self.status = Status.FAILURE

        self.logger.log_event( "End Solver" )

    def phase_4_Execute_Action( self ):
        """ Attempt to execute the first action in the symbolic plan """
        
        btr = BT_Runner( self.action, self.world, 20.0, 30.0 )
        btr.setup_BT_for_running()

        while not btr.p_ended():
            btr.tick_once()
            if (btr.status == Status.FAILURE):
                self.status = Status.FAILURE
                self.logger.log_event( "Action Failure", btr.msg )
            if self.check_OOB( 1.5 ):
                self.status = Status.FAILURE
                self.logger.log_event( "Object OOB", str( self.world.full_scan_true() ) )


    def solve_task( self, maxIter = 100 ):
        """ Solve the goal """
        self.reset_state()
        i = 0

        print( "\n\n\n##### TAMP BEGIN #####\n" )

        self.logger.begin_trial()

        while (self.status != Status.SUCCESS) and (i < maxIter):

            print( f"### Iteration {i+1} ###" )
            
            i += 1

            print( f"Phase 1, {self.status} ..." )
            self.reset_beliefs() # WARNING: REMOVE FOR RESPONSIVE
            self.phase_1_Perceive( 3*_N_POSE_UPDT+1 )

            print( f"Phase 2, {self.status} ..." )
            self.phase_2_Ground_Symbols()

            if self.status in (Status.SUCCESS, Status.FAILURE):
                print( f"LOOP, {self.status} ..." )
                continue

            print( f"Phase 3, {self.status} ..." )
            self.phase_3_Plan_Task()

            # DEATH MONITOR
            if self.noSoln >= self.nonLim:
                self.logger.log_event( "SOLVER BRAINDEATH", f"Iteration {i+1}: Solver has failed {self.noSoln} time in a row!" )
                break

            if self.p_failed():
                print( f"LOOP, {self.status} ..." )
                continue

            print( f"Phase 4, {self.status} ..." )
            self.phase_4_Execute_Action()

            print()

        self.logger.end_trial(
            self.validate_goal_true( self.goal ),
            {'end_symbols' : list( self.symbols ) }
        )

        self.logger.save( "data/TAMP-Loop" )

        print( f"\n##### TAMP END with status {self.status} after iteration {i} #####\n\n\n" )




########## MAIN ####################################################################################
import os

##### Env. Settings #####
np.set_printoptions( precision = 3, linewidth = 130 )


##### Run Sim #####
if __name__ == "__main__":
    planner = ResponsiveExecutive()
    planner.set_table()
    planner.solve_task()

    if 0:
        print("\n\nExperiments DONE!!\n\n")
        duration =   3  # seconds
        freq     = 500 #440  # Hz
        os.system( 'play -nq -t alsa synth {} sine {}'.format(duration, freq) )
    
