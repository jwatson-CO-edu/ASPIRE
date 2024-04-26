import numpy as np
import open3d as o3d
from RealSense import RealSense
from magpie.ur5 import UR5_Interface
from magpie import poses

TRAJ = 1

def save_data(color_image, robot_pose, traj, index):
    o3d.io.write_image(f"./stream_data/{traj}/color_image_{index}.png", color_image)
    with open(f"./stream_data/{traj}/pose_{index}.txt", "w") as file:
        file.write(str(poses.pose_mtrx_to_vec(robot_pose)))

def main():
    robot = UR5_Interface()
    robot.start()

    real = RealSense(1.0)
    real.initConnection()

    index = 0
    try:
        while True:
            response = input("[Enter] - Capture Data, [Q] - Quit: ").strip().lower()
            if response == "q":
                break

            pcd, rgbd_image = real.getPCD()
            depth_image, color_image = rgbd_image.depth, rgbd_image.color
            robot_pose = robot.get_tcp_pose()

            save_data(color_image, robot_pose, TRAJ, index)
            index += 1

    except Exception as e:
        print(f"Erorr: {e}")

    finally:
        real.disconnect()
        # robot.stop()

if __name__ == "__main__":
    main()
