########## DEV PLAN ################################################################################
"""
[Y] Combine `ObjectBelief` and `ObjectMemory` --> `ObjectMemory`, SIMPLIFY!, 2024-04-11: Testing req'd
"""

########## INIT ####################################################################################

import numpy as np

from symbols import GraspObj, ObjectReading
from utils import ( multiclass_Bayesian_belief_update, get_confusion_matx, get_confused_class_reading, 
                    DataLogger, )
from PB_BlocksWorld import PB_BlocksWorld
from env_config import ( _BLOCK_SCALE, _N_CLASSES, _CONFUSE_PROB, _NULL_NAME, _NULL_THRESH, 
                         _BLOCK_NAMES, _VERBOSE )


########## HELPER FUNCTIONS ########################################################################

def d_between_obj_poses( obj1, obj2 ):
    """ Calculate the translation between poses in the same frame """
    return np.linalg.norm( np.subtract( obj1.pose[:3], obj2.pose[:3] ) )


def sorted_obj_labels( obj ):
    """ Get the label dist keys in a PREDICTABLE ORDER """
    # WARNING: THIS FUNCTION BECOMES NECESSARY *AS SOON AS* GLOBAL LABLES ARE **NOT** FIXED!
    rtnLst = list( obj.labels.keys() )
    rtnLst.sort()
    return rtnLst


def extract_dct_values_in_order( dct, keyLst ):
    """ Get the `dct` values in the order specified in `keyLst` """
    rtnLst = []
    for k in keyLst:
        if k in dct:
            rtnLst.append( dct[k] )
    return rtnLst


def extract_class_dist_sorted_by_key( obj ):
    """ Get the discrete class distribution, sorted by key name """
    return np.array( extract_dct_values_in_order( obj.labels, sorted_obj_labels( obj ) ) )


def extract_class_dist_in_order( obj, order = _BLOCK_NAMES ):
    """ Get the discrete class distribution, in order according to environment variable """
    return np.array( extract_dct_values_in_order( obj.labels, order ) )



########## BELIEFS #################################################################################

class ObjectMemory:
    """ Attempt to maintain recent and constistent object beliefs based on readings from the vision system """

    def reset_beliefs( self ):
        """ Remove all references to the beliefs, then erase the beliefs """
        self.beliefs = []


    def __init__( self ):
        """ Set belief containers """
        self.reset_beliefs()

    
    def accum_evidence_for_belief( self, evidence, belief ):
        """ Use Bayesian multiclass update on `belief`, destructive """
        evdnc = extract_class_dist_in_order( evidence )
        prior = extract_class_dist_in_order( belief   )
        keys  = _BLOCK_NAMES
        pstrr = multiclass_Bayesian_belief_update( 
            get_confusion_matx( _N_CLASSES, confuseProb = _CONFUSE_PROB ), 
            prior, 
            evdnc 
        )
        for i, key in enumerate( keys ):
            belief.labels[ key ] = pstrr[i]


    def integrate_one_reading( self, objReading, maxRadius = 3.0*_BLOCK_SCALE ):
        """ Fuse this belief with the current beliefs """
        relevant = False

        # 1. Determine if this belief provides evidence for an existing belief
        dMin     = 1e6
        belBest  = None
        for belief in self.beliefs:
            d = d_between_obj_poses( objReading, belief )
            if (d < maxRadius) and (d < dMin):
                dMin     = d
                belBest  = belief
                relevant = True

        if relevant:
            belBest.visited = True
            self.accum_evidence_for_belief( objReading, belBest )

        # 2. If this evidence does not support an existing belief, it is a new belief
        else:
            self.beliefs.append( objReading.copy() )

        # N. Return whether the reading was relevant to an existing belief
        return relevant
    

    def integrate_null( self, belief ):
        """ Accrue a non-observation """
        labels = get_confused_class_reading( _NULL_NAME, _CONFUSE_PROB, _BLOCK_NAMES )
        cnfMtx = get_confusion_matx( _N_CLASSES, _CONFUSE_PROB )
        priorB = [ belief.labels[ label ] for label in _BLOCK_NAMES ] 
        evidnc = [ labels[ label ] for label in _BLOCK_NAMES ]
        updatB = multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc )
        belief.labels = {}
        for i, name in enumerate( _BLOCK_NAMES ):
            belief.labels[ name ] = updatB[i]
    

    def unvisit_beliefs( self ):
        """ Set visited flag to False for all beliefs """
        for belief in self.beliefs:
            belief.visited = False


    def erase_dead( self ):
        """ Erase all beliefs and cached symbols that no longer have relevancy """
        retain = []
        for belief in self.beliefs:
            if belief.labels[ _NULL_NAME ] < _NULL_THRESH:
                retain.append( belief )
            elif _VERBOSE:
                print( f"{str(belief)} DESTROYED!" )
        self.beliefs = retain


    def decay_beliefs( self ):
        """ Destroy beliefs that have accumulated too many negative indications """
        for belief in self.beliefs:
            if not belief.visited:
                self.integrate_null( belief )
        self.erase_dead()
        self.unvisit_beliefs()


    def belief_update( self, evdncLst ):
        """ Gather and aggregate evidence """

        ## Integrate Beliefs ##
        cNu = 0
        cIn = 0
        self.unvisit_beliefs()
        for objEv in evdncLst:
            if self.integrate_one_reading( objEv ):
                cIn += 1
            else:
                cNu += 1

        if _VERBOSE:
            if (cNu or cIn):
                print( f"\t{cNu} new object beliefs this iteration!" )
                print( f"\t{cIn} object beliefs updated!" )
            else:
                print( f"\tNO belief update!" )
        
        ## Decay Irrelevant Beliefs ##
        self.decay_beliefs()
        
        if _VERBOSE:
            print( f"Total Beliefs: {len(self.beliefs)}" )


    def most_likely_objects( self, N = 1 ):
        """ Get the `N` most likely combinations of object classes """
        ## Init ##
        comboList = [ [1.0,[],], ]
        ## Generate all class combinations with joint probabilities ##
        for bel in self.beliefs:
            nuCombos = []
            for combo_i in comboList:
                for label_j, prob_j in bel.labels.items():
                    prob_ij = combo_i[0] * prob_j
                    objc_ij = GraspObj( self, label = label_j, pose = np.array( bel.pose ) )
                    nuCombos.append( [prob_ij, combo_i[1]+[objc_ij,],] )
            comboList = nuCombos
        ## Sort all class combinations with decreasing probabilities ##
        comboList.sort( key = (lambda x: x[0]), reverse = True )
        ## Return top combos ##
        if N == 1:
            return comboList[0][1]
        elif N > 1:
            rtnCombos = []
            for i in range(N):
                rtnCombos.append( comboList[i][1] )
            return rtnCombos
        else:
            return list()



########## BASELINE PLANNER ########################################################################

class BaselineTAMP:
    """ Basic TAMP loop against which the Method is compared """

    ##### Init ############################################################

    def reset_beliefs( self ):
        """ Erase belief memory """
        self.memory = ObjectMemory() # Distributions over objects


    def __init__( self, world = None ):
        """ Create a pre-determined collection of poses and plan skeletons """
        self.reset_beliefs()
        self.world  = world if (world is not None) else PB_BlocksWorld()
        self.logger = DataLogger()
        # DEATH MONITOR
        self.noSoln =  0
        self.nonLim = 10


    def perceive_scene( self ):
        """ Integrate one noisy scan into the current beliefs """
        self.memory.belief_update( self.world.full_scan_noisy() )


########## MAIN ####################################################################################
if __name__ == "__main__":
    planner = BaselineTAMP()
    objs    = planner.world.full_scan_noisy()

    for obj in objs:
        print( obj )