import rtde_control
import rtde_receive
from Motor_Code import Motors
import UR5_Interface as ur
import RealSense as real
import ObjectDetection as ob
import TaskPlanner as tp
import Block as bl
import TaskPlanner as tp
import Block as bl
import numpy as np
import spatialmath as sm
import copy
import time
import math

try:
    robotIP = "192.168.0.6"
    con = rtde_control.RTDEControlInterface(robotIP)
    rec = rtde_receive.RTDEReceiveInterface(robotIP)
    servoPort = "/dev/ttyACM0"
    gripperController = Motors(servoPort)
    gripperController.torquelimit(600) # used to be 600
    gripperController.speedlimit(100)
    ur = ur.UR5_Interface()
    ur.gripperController = gripperController
    try:
        ur.c = con
        ur.r = rec
        ur.gripperController = gripperController
    except Exception as e:
        raise(e)
    else:
        print("UR5 + Gripper Interface Established")
    real = real.RealSense()
    real.initConnection()
    try:
        detector = ob.ObjectDetection(real,None,moveRelative = True)
    except Exception as e:
        detector.real.pipe.stop()
        raise(e)
    urPose = ur.getPose()
    pcd,rgbdImage = detector.real.getPCD()
    depthImage,colorImage = rgbdImage.depth,rgbdImage.color
    blocks = detector.getBlocksFromImages(colorImage,depthImage,urPose,display = True)
    
    planner = tp.TaskPlanner(blocks)
    goalDict = {"on":[("blueBlock","yellowBlock")]}
    steps = planner.generatePlan(goalDict)
    print(steps)
    for block in blocks:
        print(f"{block.name} - {list(block.gripperFrameCoords)}")
    
    # urPose = ur.getPose()
    # pcd,rgbdImage = detector.real.getPCD()
    # depthImage,colorImage = rgbdImage.depth,rgbdImage.color
    # blocks = detector.getBlocksFromImages(colorImage,depthImage,urPose,display = True)

    # planner = tp.TaskPlanner(blocks)
    # goalDict = {"on":[("blueBlock","yellowBlock"),("redBlock", "yellowBlock")]}
    # steps = planner.generatePlan(goalDict)
    # print(steps)
    # for block in blocks:
    #     print(f"{block.name} - {list(block.gripperFrameCoords)}")
    
    sleepRate = 0.75
    def projectToWorldCoords(gripperFrameCoords):
        # given a goal position in gripper coords returns the displacements from the current pose in world coords
        xB,yB,zB = gripperFrameCoords
        # subtract 0.165 from block position in gripper frame to account for gripper length
        zB -= 0.155
        currentPose = ur.getPose() #SE3 Object
        # print(f"Current Pose:\n{currentPose*1000}")
        R = currentPose.R 
        pX,pY,pZ = tuple(currentPose.t)
        # xB,yB,zB here is the block position in the gripper frame which is aligned with the optoforce frame
        P_goal = np.matmul(R,np.array([xB,yB,zB]).T)  # relative position of the block in world coordinates
        print(f"P_goal:\n{P_goal}")
        dX,dY,dZ = tuple(P_goal) # quantities and directions the the gripper frame should be incremented to be centered at the block 
        return dX,dY,dZ
        
    def moveToBlock(blockPos):
        # would be better if this was block object
        # :blockPos is coord in gripper frame
        dX,dY,dZ = projectToWorldCoords(blockPos) # goalPose in world coordinates
        homePose = ur.getPose()
        dZ  += 7/1000 # up 7 mm to avoid hitting lower block
        goal1 = copy.deepcopy(homePose)
        goal1.t[2] += dZ
        ur.moveL(goal1)
        time.sleep(sleepRate)
        goal2 = goal1
        goal2.t[0] += dX
        goal2.t[1] += dY
        ur.moveL(goal2)
        time.sleep(sleepRate)
        
    def moveBackFromBlock(homePose):    
        currentPose = ur.getPose()
        # Move up 3 mm to avoid raise block to prevent friction from toppling lower block
        goal1 = copy.deepcopy(currentPose)
        goal1.t[2] -= 3/1000
        ur.moveL(goal1)
        time.sleep(sleepRate)
        currentPose = ur.getPose()
        dX,dY,dZ = tuple(homePose.t - currentPose.t)
        # Move in the XY Plane then Z Axis
        goal2 = copy.deepcopy(currentPose)
        goal2.t[0] += dX
        goal2.t[1] += dY
        ur.moveL(goal2)
        time.sleep(sleepRate)
        # Move in Z Axis back to home
        goal3 = copy.deepcopy(goal2)
        goal3.t[2] += dZ
        ur.moveL(goal3)
        time.sleep(sleepRate)

    goalBlock = blocks[1]
    blockLength = 0.02
    releaseCoords = goalBlock.gripperFrameCoords + goalBlock.getWorldFrameVerticalInGripper(blockLength)
    verticalDist = 0.02
    gX,gY,gZ = tuple(goalBlock.urPose.t)
    res = np.matmul(goalBlock.urPose.R,(sm.SE3.Trans([gX,gY,gZ+verticalDist]).t - goalBlock.urPose.t))
    # print(f"res: {projectToWorldCoords(res)} ")
    # ur.openGripper() # Open gripper
    # ur.testRoutine()
    homePose = ur.getPose()
    i = 0
    x_mod = 0.0
    y_mod = 0.0
    z_mod = 0.0    
    for step in steps:
        # Grasp and Move Home Step
        grabPos,releasePos = step
        # releasePos[1] = releasePos[1] + 0.02 + i
        # releasePos[2] = releasePos[2] + 0.00
        print("releasePos[1]: ", releasePos[1]) # 0.02654946393066391
        print("releasePos[1] plus 2.5cm: ", releasePos[1] + 0.025) # 0.05154946393066391
        releasePos[1] = releasePos[1] + 0.02 + y_mod
        releasePos[2] = releasePos[2]
        print("This is the releasePos" , releasePos)
        print("This is the GrabPos" , grabPos)
        moveToBlock(grabPos) 
        print("Done moving to block")
        ur.closeGripper(9) 
        time.sleep(sleepRate)
        moveBackFromBlock(homePose)
        moveToBlock(releasePos)
        ur.openGripper()
        moveBackFromBlock(homePose)
        # i = i + .02
        y_mod += 0.02
        z_mod -= 0.015
    
    gripperController.openGripper()
    gripperController.disconnect()
    ur.c.disconnect()
    ur.r.disconnect()
    real.disconnect()
except Exception as e:
    gripperController.openGripper()
    gripperController.disconnect()
    ur.c.disconnect()
    ur.r.disconnect()
    real.disconnect()
    raise(e)