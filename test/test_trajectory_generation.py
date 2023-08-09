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
        traj = np.array(self.traj_gen.generate(1))
        np.savetxt('traj.csv', traj, delimiter=',', fmt='%.6f')

    def test_generate_z(self):
        start = np.eye(4)
        end = np.array([
            [-1, 0, 0 ,1],
            [0, -1, 0 ,0],
            [0, 0, 1 ,0.1],
            [0, 0, 0 ,1],
        ])
        self.traj_gen.set_start_trans(start)
        self.traj_gen.set_end_trans(end)
        traj = np.array(self.traj_gen.generate(1))
        np.savetxt('traj.csv', traj, delimiter=',', fmt='%.6f')
