import numpy as np

def multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc ):
    Nclass = cnfMtx.shape[0]
    priorB = np.array( priorB ).reshape( (Nclass,1,) )
    evidnc = np.array( evidnc ).reshape( (Nclass,1,) )
    P_e    = cnfMtx.dot( priorB ).reshape( (Nclass,) )
    P_hGe  = np.zeros( (Nclass,Nclass,) )
    for i in range( Nclass ):
        P_hGe[i,:] = (cnfMtx[i,:]*priorB[i,0]).reshape( (Nclass,) ) / P_e
    return P_hGe.dot( evidnc ).reshape( (Nclass,) )

Nclass = 3
cnfMtx = np.array([[0.8,0.1,0.1],
                   [0.1,0.8,0.1],
                   [0.1,0.1,0.8],])
priorB = np.array([0.6,0.2,0.2])
evidnc = np.array([0.75,0.15,0.10])

print( multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc ) )