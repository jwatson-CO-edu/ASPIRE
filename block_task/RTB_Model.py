import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3
import swift
import spatialgeometry as sg
from spatialgeometry.geom.CollisionShape import (CollisionShape,Mesh,Cylinder,Cuboid,Box,Sphere)

class RTB_Model():
    # Kinematic Model of the Robot in the Robotics Toolbox for Python (RTB)
    # Not really needed but helpful for visualization / testing
    def __init__(self):
        # Model has units in meters
        self.ur5_DH = rtb.models.DH.UR5()  # URDF model
        self.ur5_URDF = rtb.models.URDF.UR5()  # DH parameter-based model

        homeJointAngles = np.array(np.radians([53, -112, 144, -27.5, 55, 171.7]))
        self.setJointAngles(homeJointAngles)

    def setJointAngles(self, thetas):
        # thetas: 6 x 1 numpy array of joint angles (radians)
        self.ur5_DH.q = thetas
        self.ur5_URDF.q = thetas

    def initSwiftEnv(self):
        self.swiftEnv = swift.Swift()
        self.swiftEnv.launch()
        self.swiftEnv.add(self.ur5_URDF)
        endEffectorFrame = self.ur5_DH.fkine_all(self.ur5_DH.q)[-1] * SE3()
        endEffectorFrame.t[0:2] *= -1
        axes = sg.Axes(length=0.5, pose=endEffectorFrame)
        self.swiftEnv.add(axes)

    def addSwiftBox(self, pos, dims=(0.01905, 0.01905, 0.01905)):
        # Swift Model seems to have a different coordinate system than the DH model
        # The x-axis and y-axis point in the opposite direction as the DH model
        # pos is tuple of (x,y,z)
        # dims is tuple of (length,width,height)
        box = Cuboid(dims, pose=SE3(-pos[0], -pos[1], pos[2]))
        self.swiftEnv.add(box)

    def simulateSwiftRobot(self):
        T = np.eye(4)
        T[0:3, 3] = np.array([0, 0.25, 0.25]).T
        print(T)

        # put a breakpoint here and look at the object to see what it is
        #
        sol = self.ur5_URDF.ikine_LM(SE3(T))  # Solve IK to get fgoal pose
        print(sol)
        qtraj = rtb.jtraj(self.ur5_URDF.q, sol.q, 50)
        dt = 0.05
        self.swiftEnv.start_recording("ur5", 1 / dt)
        for qk in qtraj.q:
            print(qk)
            self.setJointAngles(qk)  # update robot state
            self.swiftEnv.step(dt)  # update visualization
        self.swiftEnv.stop_recording()
        # dt = 0.050

    def plotRobot(self):
        # Displays DH robot in matplotlib
        ur5 = self.ur5_DH
        env = ur5.plot(ur5.q)  # PyPlot backend

        # env.ax.scatter([0,0],[0,0],[0.4,0.45])
        # T_C = self.getCameraFrame()
        # T_C.plot(frame="C",length=0.1)
        # env.hold()

    def getGripperPose(self):
        # Returns the pose (4 x 4 Homogenous Transform as SE3 Spatial Math Object) with position in the center between the 2 gripper links
        ur5 = self.ur5_DH
        # T_N = ur5.fkine_all(ur5.q)[-1]  # T_N - end-effector frame (before optoforce/gripper)
        T_N = ur5.G(ur5.q)[-1]
        d = 0.1125  # distance between end-effector frame origin and center of gripper frame along z-axis (m)
        T_G = T_N * SE3.Tz(d)
        return T_G

    def getCameraFrame(self):
        # Robot joint angles need to be set pior <<<<< NEED TO BE SET TO WHAT?
        # Returns a SE3 Spatial Math Object (4 x 4 Homogenous Transform) corresponding to the robot's camera frame
        ur5 = self.ur5_DH
        T_N = ur5.fkine_all(ur5.q)[-1]  # T_N - end-effector frame (before optoforce/gripper)
        d = 0.292  # distance between end-effector frame origin and center of camera frame along z-axis (m)
        P_C = np.array([0, 0, d])  # Translation from frame T_N to origin of camera frame (m)
        theta = np.radians(90)  # Rotation about the Z-axis between the camera frame and end-effector frame
        T_C = T_N * SE3.Tz(d) * SE3.Rz(theta)  # Camera coordinate frame
        return T_C