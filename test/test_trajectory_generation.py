from unittest import TestCase
from trajectory_planning.trajectory_generation import Trajectory
import numpy as np

class TestTrajectory(TestCase):


    def setUp(self) -> None:
        super().setUp()

        self.traj_gen = Trajectory()

    def test_generate(self):
        start = np.eye(4)
        end = np.array([
            [-1, 0, 0 ,0],
            [0, -1, 0 ,0],
            [0, 0, 1 ,1],
            [0, 0, 0 ,1],
        ])
        self.traj_gen.set_start_trans(start)
        self.traj_gen.set_end_trans(end)
        traj = np.array(self.traj_gen.generate(1))
        np.savetxt('traj.csv', traj, delimiter=',', fmt='%.6f')


    def test_generate_x(self):
        start = np.eye(4)
        end = np.array([
            [-1, 0, 0 ,1],
            [0, -1, 0 ,0],
            [0, 0, 1 ,0.1],
            [0, 0, 0 ,1],
        ])
        self.traj_gen.set_start_trans(start)
        self.traj_gen.set_end_trans(end)
        traj = np.array(self.traj_gen.generate(5))
        np.savetxt('traj.csv', traj, delimiter=',', fmt='%.6f')

    def test_generate_z(self):
        start = np.eye(4)
        end = np.array([
            [-1, 0, 0 ,0],
            [0, -1, 0 ,0],
            [0, 0, 1 ,1],
            [0, 0, 0 ,1],
        ])
        self.traj_gen.set_start_trans(start)
        self.traj_gen.set_end_trans(end)
        traj = np.array(self.traj_gen.generate(30))
        np.savetxt('traj.csv', traj, delimiter=',', fmt='%.6f')

    def test_generate_trajectories(self):
        start = np.eye(4)
        goals = [
            np.array([
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [-1, 0, 0, 0.05],
                [0, 0, 0, 1],
            ]),
            np.array([
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [-1, 0, 0, 0.025],
                [0, 0, 0, 1],
            ]),
            np.array([
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [-1, 0, 0, 0.1],
                [0, 0, 0, 1],
            ]),
            np.array([
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0.5],
                [0, 0, 0, 1],
            ]),
        ]
        gripper_states = [0, 0, 1, 1]

        times = [5, 1, 2, 5]
        traj = np.array(self.traj_gen.generate_trajectories(start, goals, times, gripper_states))
        np.savetxt('traj.csv', traj, delimiter=',', fmt='%.6f')

    def test_generate_pickup_dropoff_trajectory(self):
        Tse_init = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.5],
            [0, 0, 1, 0.5],
            [0, 0, 0, 1.0],
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
            [ 0, 0, 1, 0.0],
            [ 0, -1, 0, 0.0],
            [ 1, 0, 0, 0.1],
            [ 0, 0, 0, 1.0],
        ])

        Tce_grasp = np.array([
            [ 0, 0, 1, 0.0],
            [ 0, -1, 0, 0.0],
            [ 1, 0, 0, 0.025],
            [ 0, 0, 0, 1.0],
        ])

        T1 = np.matmul(Tsc_init, Tce_standoff)    # above block pre
        T2 = np.matmul(Tsc_init, Tce_grasp)       # move grasp
        T3 = np.array(T2)                         # grasp
        T4 = np.array(T1)                         # above block post
        T5 = np.matmul(Tsc_final, Tce_standoff)   # above block drop-off pre
        T6 = np.matmul(Tsc_final, Tce_grasp)      # move drop-off
        T7 = np.array(T6)                         # release
        T8 = np.array(T5)                         # move drop-off post

        goals          = [T1, T2, T3, T4, T5, T6, T7, T8]
        times          = [10, 5,  3,  5,  20, 5,  3,  8]
        gripper_states = [0 , 0,  1,  1,  1,  1,  0,  0]

        traj = np.array(self.traj_gen.generate_trajectories(Tse_init, goals, times, gripper_states))
        np.savetxt('traj.csv', traj, delimiter=',', fmt='%.6f')




