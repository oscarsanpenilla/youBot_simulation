from unittest import TestCase
from kinematics.odometry import FourWheeledMecanumOdometry
import numpy as np


class TestFourWheeledMecanumOdometry(TestCase):

    def setUp(self) -> None:
        super().setUp()
        l = 0.47 / 2
        w = 0.3 / 2
        r = 0.0475
        q_init = [0, 0, 0]
        self.robot = FourWheeledMecanumOdometry(r, w, l, q_init)

    def test_get_new_base_config(self):
        next_q = self.robot.calc_new_base_config([0, 0, 0, 0], timestep=1)
        self.assertListEqual(next_q, [0] * 3)

        next_q = self.robot.calc_new_base_config([10, 10, 10, 10], timestep=1)
        phi, x, y = next_q[0], next_q[1], next_q[2]
        self.assertEqual(phi, 0.0)
        self.assertEqual(x, 0.475)
        self.assertEqual(y, 0.0)

        next_q = self.robot.calc_new_base_config([-10, 10, -10, 10], timestep=1)
        phi, x, y = next_q[0], next_q[1], next_q[2]
        self.assertEqual(phi, 0.0)
        self.assertEqual(x, 0.0)
        self.assertEqual(y, 0.475)

        next_q = self.robot.calc_new_base_config([-10, 10, 10, -10], timestep=1)
        phi, x, y = next_q[0], next_q[1], next_q[2]
        self.assertEqual(np.round(phi, 3), 1.234)
        self.assertEqual(x, 0.0)
        self.assertEqual(y, 0.0)

    def test_joint_speed_limit(self):
        self.robot.joint_max_speed = 5
        next_q = self.robot.calc_new_base_config([10, 10, 10, 10], timestep=1)
        phi, x, y = next_q[0], next_q[1], next_q[2]
        self.assertEqual(phi, 0.0)
        self.assertEqual(x, 0.475 / 2)
        self.assertEqual(y, 0.0)

        next_q = self.robot.calc_new_base_config([-10, 10, 10, -10], timestep=1)
        phi, x, y = next_q[0], next_q[1], next_q[2]
        self.assertEqual(np.round(phi, 3), 1.234 / 2.)
        self.assertEqual(x, 0.0)
        self.assertEqual(y, 0.0)
