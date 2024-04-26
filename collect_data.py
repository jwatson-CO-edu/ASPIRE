import numpy as np
import open3d as o3d
import time
import threading
from RealSense import RealSense
from magpie.ur5 import UR5_Interface
from magpie import poses

TRAJ = 1

def save_data(color_image, robot_pose, traj, index):
    o3d.io.write_image(f"./stream_data/{traj}/color_image_{index}.png", color_image)
    with open(f"./stream_data/{traj}/pose_{index}.txt", "w") as file:
        file.write(str(poses.pose_mtrx_to_vec(robot_pose)))

def capture_loop(robot, real, stop_event):
    index = 0
    while not stop_event.is_set():
        pcd, rgbd_image = real.getPCD()
        depth_image, color_image = rgbd_image.depth, rgbd_image.color
        robot_pose = robot.get_tcp_pose()

        save_data(color_image, robot_pose, TRAJ, index)
        index += 1
        time.sleep(2)

def main():
    robot = UR5_Interface()
    robot.start()

    real = RealSense(1.0)
    real.initConnection()

    initial = np.array([-np.pi / 2.0, -np.pi / 2.6, -np.pi / 2.2, -np.pi * 4.0 / 2.8, -np.pi / 2.0, 0])
    start = Move_Q(initial, name = None, ctrl = robot, rotSpeed = 1.05, rotAccel = 1.4, asynch = True)

    stop_event = threading.Event()
    capture_thread = threading.Thread(target = capture_loop, args = (robot, real, stop_event))
    capture_thread.start()

    try:
        while True:
            if input().strip().lower() == 'q':
                stop_event.set()
                break

    except Exception as e:
        print(f"Erorr: {e}")

    finally:
        capture_thread.join()
        real.disconnect()
        # robot.stop()

if __name__ == "__main__":
    main()
