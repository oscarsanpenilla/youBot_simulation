from typing import List

from kinematics.forward_kin import ForwardKin
from kinematics.odometry import FourWheeledMecanumOdometry


class Kinematics:
    def __init__(self, base_init_config, arm_init_config, wheel_angles):
        l = 0.47 / 2
        w = 0.3 / 2
        r = 0.0475
        self.base_odom = FourWheeledMecanumOdometry(r, w, l, base_init_config, wheel_angles)
        self.arm_kin = ForwardKin(init_q=arm_init_config)

    def next_state(self, wheel_speeds, joint_speeds, timestep: float) -> List[float]:
        new_base_config = self.base_odom.update_base_config(wheel_speeds, timestep)
        new_wheel_angles = self.base_odom.th_wheel_curr
        new_arm_config = self.arm_kin.update_arm_config(joint_speeds, timestep)
        return new_base_config + new_arm_config + new_wheel_angles
