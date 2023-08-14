from unittest import TestCase
import numpy as np
from control.control import Control
from kinematics.kinematics import Kinematics

np.set_printoptions(precision=3, suppress=True, linewidth=500)

class TestControl(TestCase):
    def setUp(self) -> None:
        super().setUp()
        q_base_init = [0, 0, 0]
        q_arm_init = [0, 0, 0.2, -1.6, 0]
        wheel_angles = [0] * 4
        self.robot_kin = Kinematics(q_base_init, q_arm_init, wheel_angles)
        self.controller = Control(self.robot_kin)

    def almost_equal_lists(self, list1, list2, tol=1):
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, places=tol)

    def test_feedback_control(self):
        dt = 0.01
        Xd = np.array([
            [ 0, 0, 1, 0.5],
            [ 0, 1, 0, 0  ],
            [-1, 0, 0, 0.5],
            [ 0, 0, 0, 1  ]
        ])
        Xd_next = np.array([
            [ 0, 0, 1, 0.6],
            [ 0, 1, 0, 0.0],
            [-1, 0, 0, 0.3],
            [ 0, 0, 0, 1]
        ])

        control_output = self.controller.feedback_control(dt, Xd, Xd_next, kp=0, ki=0, print_enabled=True)
        expect_control_output = [157.17, 157.17, 157.17, 157.17, -0., -652.887, 1398.59, -745.702, 0.]
        self.almost_equal_lists(control_output, expect_control_output)

        control_output = self.controller.feedback_control(dt, Xd, Xd_next, kp=1, ki=0, print_enabled=True)
        expect_control_output = [157.5, 157.5, 157.5, 157.5, 0., -654.3, 1400.9, -746.8, 0.]
        self.almost_equal_lists(control_output, expect_control_output)

    def test_feedforward_control_only(self):
        start = np.array([
            [ 0, 0, 1, 0.5],
            [ 0, 1, 0, 0. ],
            [-1, 0, 0, 0.5],
            [ 0, 0, 0, 1.0],
        ])
        end = np.array([
            [ 0, 0, 1, 0.6],
            [ 0, 1, 0, 0. ],
            [-1, 0, 0, 0.3],
            [ 0, 0, 0, 1.0],
        ])
        trajectory = [start, end]

        robot_state_list = []
        # dt = 0.01
        kp = 0
        ki = 0
        traj_size = len(trajectory)
        wheel_speeds = [0] * 4
        joint_speeds = [0] * 5
        time = np.arange(0, 1.01, 0.01)
        for dt in time:
            # if idx + 1 < traj_size:
            if True:
                q_base = self.robot_kin.base_odom.update_base_config(wheel_speeds, 0.01)
                q_arm = self.robot_kin.arm_kin.update_arm_config(joint_speeds, 0.01)
                gripper_state = 0

                Xd = trajectory[0]
                Xd_next = trajectory[1]

                ctrl_out = self.controller.feedback_control(dt, Xd, Xd_next, kp, ki)
                wheel_speeds = ctrl_out[:4]
                joint_speeds = ctrl_out[4:]

                robot_state = q_base + q_arm + self.robot_kin.base_odom.th_wheel_curr + [gripper_state]

                robot_state_list.append(robot_state)

        np.savetxt('pickup_dropoff_block_simulation.csv', np.array(robot_state_list), delimiter=',', fmt='%.6f')