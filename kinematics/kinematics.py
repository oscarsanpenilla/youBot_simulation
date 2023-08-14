from typing import List

from kinematics.forward_kin import ForwardKin
from kinematics.odometry import FourWheeledMecanumOdometry
import numpy as np
import modern_robotics as mr


class Kinematics:
    def __init__(self, base_init_config, arm_init_config, wheel_angles):
        self.l = 0.47 / 2
        self.w = 0.3 / 2
        self.r = 0.0475
        self.base_odom = FourWheeledMecanumOdometry(self.r, self.w, self.l, base_init_config, wheel_angles)
        self.arm_kin = ForwardKin(init_q=arm_init_config)

        self.Tb0 = np.array([
            [1, 0, 0, 0.1662],
            [0, 1, 0, 0.],
            [0, 0, 1, 0.0026],
            [0, 0, 0, 1.],
        ])

        self.F6 = np.zeros((6, 4))
        self.F6[2:5, :] = self.base_odom.F

    def next_state(self, wheel_speeds, joint_speeds, timestep: float) -> List[float]:
        new_base_config = self.base_odom.update_base_config(wheel_speeds, timestep)
        new_wheel_angles = self.base_odom.th_wheel_curr
        new_arm_config = self.arm_kin.update_arm_config(joint_speeds, timestep)
        return new_base_config + new_arm_config + new_wheel_angles

    def get_end_effector_transform(self):
        Tsb = self.base_odom.get_curr_base_trans()
        q_arm = self.arm_kin.get_curr_config()
        T0e = self.arm_kin.forward_kinematics(q_arm)
        Te0 = np.linalg.inv(T0e)
        Ts0 = np.matmul(Tsb, self.Tb0)
        X = np.matmul(Ts0, T0e)
        return X

    def get_curr_jacobian(self):
        T0b = np.linalg.inv(self.Tb0)
        T0e = self.arm_kin.get_curr_endeffector_transform()
        Te0 = np.linalg.inv(T0e)

        Jbase = np.matmul(mr.Adjoint(np.matmul(Te0, T0b)), self.F6)
        Jarm  = self.arm_kin.get_curr_body_jacobian()
        Je = np.hstack([Jbase, Jarm])
        return Je