import rtde_control
import rtde_receive
from Motor_Code import Motors
import UR5_Interface as ur
import RealSense as real
import ObjectDetection as ob
import TaskPlanner as tp
import Block as bl
 
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
    detector.displayWorld(pcd,blocks)
    gripperController.disconnect()
    ur.c.disconnect()
    ur.r.disconnect()
    real.disconnect()
except Exception as e:
    gripperController.disconnect()
    ur.c.disconnect()
    ur.r.disconnect()
    real.disconnect()
    raise(e)