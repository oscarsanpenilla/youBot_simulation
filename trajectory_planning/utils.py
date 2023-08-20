import numpy as np
import modern_robotics as mr

def trajectory_to_csv(trajectories: np.array, gripper_states, file_name: str = 'traj.csv'):
    trans_list = []
    trajectories_ = np.array(trajectories)

    if len(trajectories_.shape) == 3:
        trajectories_ = np.array([trajectories_])

    for idx, traj in enumerate(trajectories_):
        for trans in traj:
            rot_mat = trans[:3, :3]
            trans_mat = trans[:3, 3]
            trans_list.append(rot_mat.flatten().tolist() + trans_mat.flatten().tolist() + [gripper_states[idx]])

    np.savetxt(file_name, np.array(trans_list), delimiter=',', fmt='%.6f')


def generate_new_task_trajectory(y_tilt_angle=20):
    # Define the rotation axis for the Y-axis
    omega = np.array([0, 1, 0])
    # Define the desired rotation angle in radians
    theta = np.radians(y_tilt_angle)
    # Compute the skew-symmetric matrix corresponding to the axis-angle representation
    S = mr.VecToso3(omega * theta)
    # Compute the rotation matrix using the MatrixExp3 function
    rot_y = mr.MatrixExp3(S)
    trans_y = np.eye(4)
    trans_y[:3, :3] = rot_y

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
        [0, 0, 1, 0.185],
        [0, 0, 0, 1.0],
    ])

    Tce_standoff = np.array([
        [0, 0, 1, 0.0],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.25],
        [0, 0, 0, 1.0],
    ])
    Tce_standoff = np.matmul(Tce_standoff, trans_y)

    Tce_grasp = np.array([
        [0, 0, 1, 0.0],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.025],
        [0, 0, 0, 1.0],
    ])
    Tce_grasp = np.matmul(Tce_grasp, trans_y)

    T1 = np.matmul(Tsc_init, Tce_standoff)  # above block pre
    T2 = np.matmul(Tsc_init, Tce_grasp)  # move grasp
    T3 = np.array(T2)  # grasp
    T4 = np.array(T1)  # above block post
    T5 = np.matmul(Tsc_final, Tce_standoff)  # above block drop-off pre
    T6 = np.matmul(Tsc_final, Tce_grasp)  # move drop-off
    T7 = np.array(T6)  # release
    T8 = np.array(T5)  # move drop-off post

    goals = [T1, T2, T3, T4, T5, T6, T7, T8]
    times = [5, 2, 1, 2, 12, 4, 1, 4]
    gripper_states = [0, 0, 1, 1, 1, 1, 0, 0]
    wheel_rev_sec = 3 * 360 * np.pi / 180
    joint_rev_sec = 1.5 * 360 * np.pi / 180
    traj_max_speeds = {
        0: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        1: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        2: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        3: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        4: {
            "base": 0.0,
            "arm": 0.1 * 360 * np.pi / 180,
        },
        5: {
            "base": 0.0,
            "arm": 0.1 * 360 * np.pi / 180,
        },
        6: {
            "base": 0.0,
            "arm": 0.1 * 360 * np.pi / 180,
        },
        7: {
            "base": 0 * 360 * np.pi / 180,
            "arm": 0.1 * 360 * np.pi / 180,
        },
    }

    return start, goals, times, gripper_states, traj_max_speeds


def gen_pickup_dropoff_block_trajectory(y_tilt_angle=20):
    # Define the rotation axis for the Y-axis
    omega = np.array([0, 1, 0])
    # Define the desired rotation angle in radians
    theta = np.radians(y_tilt_angle)
    # Compute the skew-symmetric matrix corresponding to the axis-angle representation
    S = mr.VecToso3(omega * theta)
    # Compute the rotation matrix using the MatrixExp3 function
    rot_y = mr.MatrixExp3(S)
    trans_y = np.eye(4)
    trans_y[:3,:3] = rot_y

    start = np.array([
        [0, 0, 1, 0.],
        [0, 1, 0, 0.],
        [-1, 0, 0, 0.5],
        [0, 0, 0, 1.0]
    ])

    Tsc_init = np.array([
        [1, 0, 0, 1.0],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.0],
        [0, 0, 0, 1.0],
    ])

    Tsc_final = np.array([
        [0, 1, 0, 0.0],
        [-1, 0, 0, -1.0],
        [0, 0, 1, 0.0],
        [0, 0, 0, 1.0],
    ])

    Tce_standoff = np.array([
        [0, 0, 1, 0.0],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.1],
        [0, 0, 0, 1.0],
    ])
    Tce_standoff = np.matmul(Tce_standoff, trans_y)

    Tce_grasp = np.array([
        [0, 0, 1, 0.0],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.025],
        [0, 0, 0, 1.0],
    ])
    Tce_grasp = np.matmul(Tce_grasp, trans_y)

    T1 = np.matmul(Tsc_init, Tce_standoff)  # above block pre
    T2 = np.matmul(Tsc_init, Tce_grasp)  # move grasp
    T3 = np.array(T2)  # grasp
    T4 = np.array(T1)  # above block post
    T5 = np.matmul(Tsc_final, Tce_standoff)  # above block drop-off pre
    T6 = np.matmul(Tsc_final, Tce_grasp)  # move drop-off
    T7 = np.array(T6)  # release
    T8 = np.array(T5)  # move drop-off post

    goals = [T1, T2, T3, T4, T5, T6, T7, T8]
    times = [5, 2, 1, 2, 5, 2, 1, 2]
    gripper_states = [0, 0, 1, 1, 1, 1, 0, 0]
    wheel_rev_sec = 1 * 360 * np.pi / 180
    joint_rev_sec = 1.5 * 360 * np.pi / 180
    traj_max_speeds = {
        0: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        1: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        2: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        3: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        4: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        5: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        6: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
        7: {
            "base": wheel_rev_sec,
            "arm": joint_rev_sec,
        },
    }

    return start, goals, times, gripper_states, traj_max_speeds