########## INIT ####################################################################################

##### Imports #####

import os, math, time
now = time.time

from random import random

import numpy as np

from spatialmath.quaternion import UnitQuaternion
from spatialmath.base import r2q





########## HELPER FUNCTIONS ########################################################################

def origin_row_vec():
    """ Return a row vector representing the origin pose as a Position and Orientation --> [Px,Py,Pz,Ow,Ox,Oy,Oz] """
    return [0,0,0,1,0,0,0]

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
    """ [Px,Py,Pz,Ow,Ox,Oy,Oz] --> [Px,Py,Pz],[Ox,Oy,Oz,Ow] """
    posn = np.array( V[0:3] )
    ornt = np.zeros( (4,) )
    ornt[:3] = V[4:7]
    ornt[-1] = V[3]
    return posn, ornt

def homog_to_row_vec( homog ):
    """ Express a homogeneous coord as a Position and Orientation --> [Px,Py,Pz,Ow,Ox,Oy,Oz] """
    P = homog[0:3,3]
    Q = UnitQuaternion( r2q( homog[0:3,0:3] ) )
    V = np.zeros( (7,) )
    V[0:3] = P[:]
    V[3]   = Q.s
    V[4:7] = Q.v[:]
    return np.array(V)

def row_vec_to_homog( V ):
    """ Express [Px,Py,Pz,Ow,Ox,Oy,Oz] as homogeneous coordinates """
    posn, ornt = row_vec_to_pb_posn_ornt( V )
    return pb_posn_ornt_to_homog( posn, ornt )


def homog_to_pb_posn_ornt( homog ):
    """ Express a homogeneous coord as a Position and Orientation --> [Px,Py,Pz],[Ox,Oy,Oz,Ow] """
    return row_vec_to_pb_posn_ornt( homog_to_row_vec( homog ) )


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