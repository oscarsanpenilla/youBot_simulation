from unittest import TestCase
from kinematics.kinematics import Kinematics
import numpy as np


class TestKinematics(TestCase):
    def setUp(self) -> None:
        super().setUp()

        base_init_config = [0] * 3
        arm_init_config = [0] * 5
        wheel_angles = [0] * 4
        self.robot_kin = Kinematics(base_init_config, arm_init_config, wheel_angles)

    def test_next_state_move_x(self):
        states = []
        for i in range(100):
            states.append(np.append(np.array(self.robot_kin.next_state(
                [10,10,10,10], [1] * 5, 0.01)), 0))
        states = np.array(states)
        print(states)

        np.savetxt('array.csv', states, delimiter=',', fmt='%.6f')

    def test_next_state_move_y(self):
        states = []
        for i in range(100):
            states.append(np.append(np.array(self.robot_kin.next_state(
                [-10, 10, -10, 10], [1] * 5, 0.01)), 0))
        states = np.array(states)
        print(states)

        np.savetxt('array.csv', states, delimiter=',', fmt='%.6f')

    def test_next_state_rotate_in_place(self):
        states = []
        for i in range(100):
            states.append(np.append(np.array(self.robot_kin.next_state(
                [-10, 10, 10, -10], [1] * 5, 0.01)), 0))
        states = np.array(states)
        print(states)

        np.savetxt('array.csv', states, delimiter=',', fmt='%.6f')

    def test_next_state_arm_only(self):
        states = []
        for i in range(100):
            states.append(np.append(np.array(self.robot_kin.next_state(
                [0]*4, [0,0,0,0,1], 0.01)), 0))
        states = np.array(states)
        print(states)

        np.savetxt('array.csv', states, delimiter=',', fmt='%.6f')

