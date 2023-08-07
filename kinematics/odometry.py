import numpy as np
from numpy import sin, cos
import sympy as sp
from typing import List, Dict

class FourWheeledMecanumOdometry:
    def __init__(self, r, w, l, chasis_config: List[float], wheel_angles: List[float] = None):
        self.r = r
        self.w = w
        self.l = l
        self.q_init = chasis_config.copy()
        self.q_curr = self.q_init.copy()
        self._th_wheel_curr = [0] * 4 if not wheel_angles else wheel_angles
        self._d_th = None
        self._joint_max_speed = 9e9
        self._total_time = 0.0

    @property
    def th_wheel_curr(self):
        return list(self._th_wheel_curr)

    @property
    def joint_max_speed(self):
        return self._joint_max_speed

    @joint_max_speed.setter
    def joint_max_speed(self, value):
        self._joint_max_speed = value

    def reset(self):
        self.q_curr = self.q_init
        self._th_wheel_curr = np.zeros(4)
        self._total_time = 0.0

    def update_base_config(self, u: List[float], timestep: float) -> List[float]:
        self.q_curr = self.calc_new_base_config(u, timestep)
        self._th_wheel_curr += self._d_th
        return self.q_curr.copy()

    def calc_new_base_config(self, u: List[float], timestep: float) -> List[float]:
        if len(u) != 4:
            raise RuntimeError("d_theta must be of size 4")

        # Enforce joint speed limits
        control_input = np.array([np.sign(speed) * min(abs(speed), self._joint_max_speed) for speed in u])

        l = self.l
        r = self.r
        w = self.w
        phi, x, y = self.q_curr[0], self.q_curr[1], self.q_curr[2]

        F = r/4* np.array([
            [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
            [    1   ,   1    ,    1   ,   1     ],
            [   -1   ,   1    ,   -1   ,   1     ]
        ])

        self._d_th = control_input * timestep

        Vb = np.matmul(F, self._d_th)

        wbz = Vb[0]
        vbx = Vb[1]
        vby = Vb[2]

        d_qb = None
        if wbz == 0:
            d_qb = np.array([0, vbx, vby])

        else:
            d_qb = np.array([
                [wbz],
                [vbx*sin(wbz) + vby/wbz * (cos(wbz)-1)],
                [vby*sin(wbz) + vbx/wbz * (1-cos(wbz))],
            ])

        chassis_rot = np.array([
            [1, 0, 0],
            [0, cos(phi), -sin(phi)],
            [0, sin(phi), cos(phi)],
        ])
        d_q = np.matmul(chassis_rot, d_qb).flatten()

        next_q = self.q_curr + d_q
        self._total_time += timestep
        return  next_q.tolist()



