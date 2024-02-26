import numpy as np
from spatialmath import Quaternion
from spatialmath.quaternion import UnitQuaternion
from spatialmath.base import r2q


def origin_row_vec():
    """ Return a row vector representing the origin pose as a Position and Orientation --> [Px,Py,Pz,Ow,Ox,Oy,Oz] """
    return [0,0,0,1,0,0,0]


def NaN_row_vec():
    """ Return a row vector representing an error in pose calculation """
    return [float("nan") for _ in range(7)]


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


def row_vec_normd_ornt( V ):
    """ Normalize the orientation of the [Px,Py,Pz,Ow,Ox,Oy,Oz] pose """
    rtnV = [v_i for v_i in V[:3]]
    q   = Quaternion( s = V[3], v = V[4:] )
    qN  = q.unit()
    rtnV.append( qN.s )
    rtnV.extend( qN.v.tolist() )
    return np.array( rtnV )


def row_vec_to_homog( V ):
    """ Express [Px,Py,Pz,Ow,Ox,Oy,Oz] as homogeneous coordinates """
    posn, ornt = row_vec_to_pb_posn_ornt( V )
    return pb_posn_ornt_to_homog( posn, ornt )


def homog_to_pb_posn_ornt( homog ):
    """ Express a homogeneous coord as a Position and Orientation --> [Px,Py,Pz],[Ox,Oy,Oz,Ow] """
    return row_vec_to_pb_posn_ornt( homog_to_row_vec( homog ) )


def pose_covar( stddev ):
    """ Get the pose covariance """
    rtnArr = np.zeros( (7,7,) )
    for i in range(7):
        rtnArr[i,i] = (stddev[i])**2
    return rtnArr


def sample_pose( pose, stddev ):
    """ Sample a pose from the present distribution, Reset on failure """
    try:
        posnSample = np.random.multivariate_normal( pose, pose_covar( stddev ) ) 
    except (np.linalg.LinAlgError, RuntimeWarning,):
        return NaN_row_vec()
    return row_vec_normd_ornt( posnSample )