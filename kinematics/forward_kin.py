from typing import List

import numpy as np
import modern_robotics as mr

class ForwardKin:
    def __init__(self, init_q: List[float]):
        self._init_q = np.array(init_q)
        self._curr_arm_th = self._init_q.copy()
        self._joint_max_speed = 9e9
        self._joint_speeds = np.array([self._joint_max_speed] * len(self._init_q))

    @property
    def joint_max_speed(self):
        return self._joint_max_speed

    @joint_max_speed.setter
    def joint_max_speed(self, value):
        self._joint_max_speed = abs(value)

    def get_new_arm_config(self, joint_speeds: List[float], timestep: float) -> List[float]:
        # Enforce joint speed limits
        j_speeds = np.array([np.sign(speed) * min(abs(speed), self._joint_max_speed) for speed in joint_speeds])

        new_angles = self._curr_arm_th + j_speeds * timestep
        return new_angles.tolist()

    def update_arm_config(self, joint_speeds: List[float], timestep: float) -> List[float]:
        self._curr_arm_th = self.get_new_arm_config(joint_speeds, timestep)
        return self._curr_arm_th.copy()


