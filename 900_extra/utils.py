########## INIT ####################################################################################

##### Imports #####

import pickle, math, time, datetime, copy
now = time.time

from random import random

import numpy as np







########## HELPER FUNCTIONS ########################################################################

def p_lst_has_nan( lst ):
    """ Does the list contain NaN? """
    for elem in lst:
        if math.isnan( elem ):
            return True
    return False

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

def diff_norm( v1, v2 ):
    """ Return the norm of the difference between the two vectors """
    return np.linalg.norm( np.subtract( v1, v2 ) )

def closest_dist_Q_to_segment_AB( Q, A, B, includeEnds = True ):
    """ Return the closest distance of point Q to line segment AB """
    l = diff_norm( B, A )
    if l <= 0.0:
        return diff_norm( Q, A )
    D = np.subtract( B, A ) / l
    V = np.subtract( Q, A )
    t = V.dot( D )
    if (t > l) or (t < 0.0):
        if includeEnds:
            return min( diff_norm( Q, A ), diff_norm( Q, B ) )
        else:
            return float("NaN")
    P = np.add( A, D*t )
    return diff_norm( P, Q ) 


def p_list_duplicates( lst ):
    """ Return True if a value appears more than once """
    s = set( lst )
    return (len( lst ) > len( s ))



########## EXPERIMENT STATISTICS ###################################################################

class DataLogger:
    """ Keep track of when experiments begin and end """

    def __init__( self ):
        """ Setup stats dict """
        self.g_BGN   = None
        self.g_RUN   = False
        self.metrics = {
            "N"     : 0,
            "pass"  : 0,
            "fail"  : 0,
            "trials": [],
        }

    def begin_trial( self ):
        """ Increment number of trials and set state """
        self.g_BGN = now()
        self.g_RUN = True
        self.metrics['N'] += 1
        self.events = []

    def log_event( self, event, msg = "" ):
        """ Log a timestamped event """
        self.events.append( (now()-self.g_BGN, event, msg,) )

    def end_trial( self, p_pass, infoDict = None ):
        """ Record makespan and trial info """
        if infoDict is None:
            infoDict = {}
        runDct = {
            "makespan" : now() - self.g_BGN,
            "result"   : p_pass,
            "events"   : list( self.events ),
        }
        self.events = []
        runDct.update( infoDict )
        self.metrics['trials'].append( runDct )
        if p_pass:
            self.metrics['pass'] += 1
        else:
            self.metrics['fail'] += 1


    def get_snapshot( self ):
        """ Copy and return the data as it currently exists """
        return copy.deepcopy( self.metrics )


    def merge_from( self, otherLogger ):
        """ Merge the metrics from another logger """
        self.metrics['N']    += otherLogger.metrics['N']
        self.metrics['pass'] += otherLogger.metrics['pass']
        self.metrics['fail'] += otherLogger.metrics['fail']
        self.metrics['trials'].extend( otherLogger.metrics['trials'] )


    def save( self, prefix = "Experiment-Data" ):
        """ Serialize recorded stats """
        fName = str( prefix ) + "__" + str( datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S') ) + ".pkl"
        with open( fName, 'wb' ) as handle:
            pickle.dump( self.metrics, handle )
        print( f"Wrote: {fName}" ) 


    def load( self, path ):
        """ De-serialize recorded stats """
        with open( path, 'rb' ) as handle:
            self.metrics = pickle.load( handle )
        return self.metrics