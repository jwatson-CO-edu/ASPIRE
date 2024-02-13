import time
from random import random

import pybullet as p
import pybullet_data

physicsClient = p.connect( p.GUI ) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath( pybullet_data.getDataPath() ) #optionally
p.setGravity( 0, 0, -10 )

planeId  = p.loadURDF( "plane.urdf" )

redBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
p.changeVisualShape( redBlock, -1, rgbaColor=[1.0, 0.0, 0.0, 1] )

ylwBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
p.changeVisualShape( ylwBlock, -1, rgbaColor=[1.0, 1.0, 0.0, 1] )

bluBlock = p.loadURDF( "cube.urdf", [ random()*3.0-1.5, random()*3.0-1.5, 0.150 ], globalScaling = 0.25  )
p.changeVisualShape( bluBlock, -1, rgbaColor=[0.0, 0.0, 1.0, 1] )

for i in range( 2000 ):
    p.stepSimulation()
    time.sleep( 1.0 / 240.0 )


p.disconnect()