from control.control import Control
from kinematics.kinematics import Kinematics
from trajectory_planning.trajectory_generation import Trajectory
import numpy as np
from numpy import cos, sin, pi
from plotter import plot_columns_from_csv
import itertools

def main():
    print("Program started...")
    q_base_init = [np.radians(10), -.2, -.2]
    q_arm_init = [1.16, 0, 0.2, -1.6, 0.1]
    wheel_angles = [0] * 4
    robot_kin = Kinematics(q_base_init, q_arm_init, wheel_angles)
    controller = Control(robot_kin)

    print("\tCalculating Trajectories...")
    start, goals, times, gripper_states, traj_max_speeds = gen_pickup_dropoff_block_trajectory()
    traj_gen = Trajectory()
    trajectory = traj_gen.generate_trajectories(start, goals, times, gripper_states)

    print("\tProcessing simulation...")
    robot_state_list = []
    error_list = []
    dt = 0.01
    # Best
    kp, ki = 0.71, 0.001
    # Overshoot
    # kp, ki = 1.4, 0.01
    # kp = 10.5
    # ki = 5
    traj_size = len(trajectory)
    wheel_speeds = [0] * 4
    joint_speeds = [0] * 5
    traj_idx_upper = list(itertools.accumulate(traj_gen.get_trajectory_sizes()))
    for idx, pose in enumerate(trajectory):
        if idx + 1 >= traj_size:
            break

        # Set joint speeds for base and arm
        traj_num = get_curr_exec_traj_num(idx, traj_idx_upper)
        robot_kin.base_odom.joint_max_speed = traj_max_speeds[traj_num]["base"]
        robot_kin.arm_kin.joint_max_speed = traj_max_speeds[traj_num]["arm"]

        q = robot_kin.next_state(wheel_speeds, joint_speeds, dt)
        gripper_state = int(trajectory[idx][-1])

        Xd = array_to_transformation_matrix(trajectory[idx][:12])
        Xd_next = array_to_transformation_matrix(trajectory[idx + 1][:12])

        ctrl_out = controller.feedback_control(dt, Xd, Xd_next, kp, ki)
        wheel_speeds = ctrl_out[:4]
        joint_speeds = ctrl_out[4:]

        robot_state = q + [gripper_state]
        error = controller.get_curr_error()

        robot_state_list.append(robot_state)
        error_list.append(error)

    print("\tWriting csv file simulation...")
    T1_size = int(times[0]//0.01)
    np.savetxt('CoppeliaSim.csv', np.array(robot_state_list), delimiter=',', fmt='%.4f')

    print("\tWriting csv file error...")
    np.savetxt('error.csv', np.array(error_list[:T1_size]), delimiter=',', fmt='%.4f')

    print("\tGenerating Plots...")
    plot_columns_from_csv('error.csv', 'output_plots.png')

    print("End")


def get_curr_exec_traj_num(idx, traj_idx_upper):
    traj_num = None
    if 0 <= idx < traj_idx_upper[0]:
        traj_num = 0
    elif traj_idx_upper[0] <= idx < traj_idx_upper[1]:
        traj_num = 1
    elif traj_idx_upper[1] <= idx < traj_idx_upper[2]:
        traj_num = 2
    elif traj_idx_upper[2] <= idx < traj_idx_upper[3]:
        traj_num = 3
    elif traj_idx_upper[3] <= idx < traj_idx_upper[4]:
        traj_num = 4
    elif traj_idx_upper[4] <= idx < traj_idx_upper[5]:
        traj_num = 5
    elif traj_idx_upper[5] <= idx < traj_idx_upper[6]:
        traj_num = 6
    elif traj_idx_upper[6] <= idx < traj_idx_upper[7]:
        traj_num = 7
    return traj_num

def rotation_about_Y_axis(angle: float):
    th = np.radians(angle)
    return np.array([
        [cos(th), 0, -sin(th), 0],
        [0, 1, 0, 0],
        [sin(th), 0, cos(th), 0],
        [0, 0, 0, 1]
    ])

def gen_pickup_dropoff_block_trajectory():
    rot_y = rotation_about_Y_axis(-20)
    start = np.array([
        [ 0, 0, 1, 0.],
        [ 0, 1, 0, 0.],
        [-1, 0, 0, 0.5],
        [ 0, 0, 0, 1.0]
    ])

    Tsc_init = np.array([
        [1, 0, 0, 1.0],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.0],
        [0, 0, 0, 1.0],
    ])

    Tsc_final = np.array([
        [ 0, 1,  0,  0.0],
        [-1, 0,  0, -1.0],
        [ 0, 0,  1,  0.0],
        [ 0, 0,  0,  1.0],
    ])

    Tce_standoff = np.array([
        [ 0,  0, 1, 0.0],
        [ 0,  1, 0, 0.0],
        [-1,  0, 0, 0.1],
        [ 0,  0, 0, 1.0],
    ])
    Tce_standoff = np.matmul(Tce_standoff, rot_y)

    Tce_grasp = np.array([
        [ 0,  0, 1, 0.0],
        [ 0,  1, 0, 0.0],
        [-1,  0, 0, 0.025],
        [ 0,  0, 0, 1.0],
    ])
    Tce_grasp = np.matmul(Tce_grasp, rot_y)

    T1 = np.matmul(Tsc_init, Tce_standoff)  # above block pre
    T2 = np.matmul(Tsc_init, Tce_grasp)  # move grasp
    T3 = np.array(T2)  # grasp
    T4 = np.array(T1)  # above block post
    T5 = np.matmul(Tsc_final, Tce_standoff)  # above block drop-off pre
    T6 = np.matmul(Tsc_final, Tce_grasp)  # move drop-off
    T7 = np.array(T6)  # release
    T8 = np.array(T5)  # move drop-off post

    # goals = [T1, T2]
    # times = [10,5]
    # gripper_states = [0,0]
    goals = [T1, T2, T3, T4, T5, T6, T7, T8]
    times = [10, 5, 3, 5, 20, 5, 3, 8]
    gripper_states = [0, 0, 1, 1, 1, 1, 0, 0]
    traj_max_speeds = {
        0 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        1 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        2 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        3 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        4 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        5 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        6 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        7 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
    }

    return start, goals, times, gripper_states, traj_max_speeds


def generate_new_task_trajectory():
    rot_y = rotation_about_Y_axis(-20)
    start = np.array([
        [0.1699, 0, 0.9854, 0.22],
        [0, 1, 0, 0.0],
        [-0.9854, 0, 0.1699, 0.4713],
        [0, 0, 0, 1.0],
    ])

    Tsc_init = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, -0.5],
        [0, 0, 1, 0.0],
        [0, 0, 0, 1.0],
    ])

    Tsc_final = np.array([
        [-1, 0, 0, 0.2],
        [0, -1, 0, -0.5],
        [0, 0, 1, 0.175],
        [0, 0, 0, 1.0],
    ])

    Tce_standoff = np.array([
        [0, 0, 1, 0.0],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.25],
        [0, 0, 0, 1.0],
    ])
    Tce_standoff = np.matmul(Tce_standoff, rot_y)

    Tce_grasp = np.array([
        [0, 0, 1, 0.0],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.025],
        [0, 0, 0, 1.0],
    ])
    Tce_grasp = np.matmul(Tce_grasp, rot_y)

    T1 = np.matmul(Tsc_init, Tce_standoff)  # above block pre
    T2 = np.matmul(Tsc_init, Tce_grasp)  # move grasp
    T3 = np.array(T2)  # grasp
    T4 = np.array(T1)  # above block post
    T5 = np.matmul(Tsc_final, Tce_standoff)  # above block drop-off pre
    T6 = np.matmul(Tsc_final, Tce_grasp)  # move drop-off
    T7 = np.array(T6)  # release
    T8 = np.array(T5)  # move drop-off post

    goals = [T1, T2, T3, T4, T5, T6, T7, T8]
    times = [10, 5, 3, 5, 20, 5, 3, 8]
    gripper_states = [0, 0, 1, 1, 1, 1, 0, 0]
    traj_max_speeds = {
        0 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        1 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        2 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        3 : {
            "base" : 3 * 360 * np.pi/180,
            "arm"  : 1.5 * 360 * np.pi/180,
        },
        4 : {
            "base" : 0 * 360 * np.pi/180,
            "arm"  : 0.1 * 360 * np.pi/180,
        },
        5 : {
            "base" : 0 * 360 * np.pi/180,
            "arm"  : 0.1 * 360 * np.pi/180,
        },
        6 : {
            "base" : 0 * 360 * np.pi/180,
            "arm"  : 0.1 * 360 * np.pi/180,
        },
        7 : {
            "base" : 0 * 360 * np.pi/180,
            "arm"  : 0.1 * 360 * np.pi/180,
        },
    }

    return start, goals, times, gripper_states, traj_max_speeds


def array_to_transformation_matrix(arr):
    arr = np.array(arr)
    if arr.size != 12:
        raise ValueError('Input array must be of size 12')

    rotation_matrix = arr[:9].reshape(3, 3)
    translation_vector = arr[9:]

    # Create a 4x4 homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


if __name__ == '__main__':
    main()

