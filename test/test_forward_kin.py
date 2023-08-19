from unittest import TestCase

import numpy as np

from kinematics.arm_kinematics import ArmKinematics


class TestForwardKin(TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.arm = ArmKinematics([0, 0, 0, 0, 0])

    def test_get_new_arm_config(self):
        new_config = self.arm.get_new_arm_config([0] * 5, 0)
        self.assertListEqual(new_config, [0] * 5)

        new_config = self.arm.get_new_arm_config([1, 0, 0, 0, 0], 1)
        self.assertListEqual(new_config, [1, 0, 0, 0, 0])

        new_config = self.arm.get_new_arm_config([1] * 5, 1)
        self.assertListEqual(new_config, [1, 1, 1, 1, 1])

        new_config = self.arm.get_new_arm_config([1] * 5, 0.5)
        self.assertListEqual(new_config, [0.5] * 5)

    def test_joint_speed_limit(self):
        self.arm.joint_max_speed = 0.5
        new_config = self.arm.get_new_arm_config([1] * 5, 1)
        self.assertListEqual(new_config, [0.5] * 5)

        self.arm.joint_max_speed = 4
        new_config = self.arm.get_new_arm_config([1, 2, 3, 4, 5], 1)
        self.assertListEqual(new_config, [1, 2, 3, 4, 4])

        self.arm.joint_max_speed = 2
        new_config = self.arm.get_new_arm_config([1, 2, 3, 4, 5], 1)
        self.assertListEqual(new_config, [1, 2, 2, 2, 2])

        self.arm.joint_max_speed = 2
        new_config = self.arm.get_new_arm_config([-1, -2, -3, -4, -5], 1)
        self.assertListEqual(new_config, [-1, -2, -2, -2, -2])

    def test_forward_kinematics(self):
        T0e = self.arm.forward_kinematics([0]*5)
        print(T0e)
