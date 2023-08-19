from control.control import Control
from kinematics.mobile_manipulator_kin import MobileManipulatorKin
from trajectory_planning.screw_axis_trajectory_generator import ScrewAxisTrajectoryGenerator
from trajectory_planning.utils import gen_pickup_dropoff_block_trajectory, generate_new_task_trajectory, \
    trajectory_to_csv

import numpy as np
from plotter import plot_columns_from_csv
from enum import Enum

class Task(Enum):
    PICKUP_DROP_OFF = 0
    PLATFORM = 1

def runscript(kp, ki, task: Task):
    print("Program started...")
    q_base_init = [np.radians(10), -.2, -.2]
    q_arm_init = [1.16, 0, 0.2, -1.6, 0.1]
    wheel_angles = [0] * 4
    robot_kin = MobileManipulatorKin(q_base_init, q_arm_init, wheel_angles)
    controller = Control(robot_kin)

    print("\tCalculating Trajectories...")
    traj_gen = ScrewAxisTrajectoryGenerator()
    start, goals, times, gripper_states, traj_max_speeds = None, None, None, None, None
    if task == Task.PICKUP_DROP_OFF:
        start, goals, times, gripper_states, traj_max_speeds = gen_pickup_dropoff_block_trajectory()
    elif task == Task.PLATFORM:
        start, goals, times, gripper_states, traj_max_speeds = generate_new_task_trajectory()
    trajectories = traj_gen.generate_trajectories(start, goals, times, gripper_states)

    print("\tWriting Trajectories csv file...")
    trajectory_to_csv(trajectories, gripper_states, 'traj.csv')

    print("\tProcessing simulation...")
    robot_state_list = []
    error_list = []
    dt = 0.01

    wheel_speeds = [0] * 4
    joint_speeds = [0] * 5
    for traj_num, traj in enumerate(trajectories):
        # Set joint speeds for base and arm
        robot_kin.base_odom.joint_max_speed = traj_max_speeds[traj_num]["base"]
        robot_kin.arm_kin.joint_max_speed = traj_max_speeds[traj_num]["arm"]

        traj_size = len(traj)
        for idx, _ in enumerate(traj):

            if idx + 1 >= traj_size:
                break

            q = robot_kin.next_state(wheel_speeds, joint_speeds, dt)
            gripper_state = gripper_states[traj_num]

            Xd = traj[idx]
            Xd_next = traj[idx + 1]

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


def main():

    # Running Pickup drop-off block task with best control constants
    kp, ki = 0.5, 0.01
    runscript(kp, ki, Task.PICKUP_DROP_OFF)

    # # Running Pickup drop-off block task with overshoot control constants
    # kp, ki = 2.73, 0.01
    # runscript(kp, ki, Task.PICKUP_DROP_OFF)
    #
    # # Running Platform task
    # kp, ki = 0.5, 0.01
    # runscript(kp, ki, Task.PLATFORM)

if __name__ == '__main__':
    main()

