import numpy as np
from numpy import sin, cos
import sympy as sp
from typing import List, Dict

class FourWheeledMecanumOdometry:
    def __init__(self, r, w, l, q_init: List[float]):
        self.r = r
        self.w = w
        self.l = l
        self.q_init = q_init.copy()
        self.q_curr = self.q_init.copy()

    def update_base_config(self, u: List[float]) -> List[float]:
        self.q_curr = self.calc_new_base_config(u)
        return self.q_curr.copy()

    def calc_new_base_config(self, u: List[float]) -> List[float]:
        if len(u) != 4:
            raise RuntimeError("d_theta must be of size 4")

        l = self.l
        r = self.r
        w = self.w
        phi, x, y = self.q_curr[0], self.q_curr[1], self.q_curr[2]

        F = r/4* np.array([
            [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
            [    1   ,   1    ,    1   ,   1     ],
            [   -1   ,   1    ,   -1   ,   1     ]
        ])

        control_input = np.array(u).T
        Vb = np.matmul(F, control_input)

        wbz = Vb[0]
        vbx = Vb[1]
        vby = Vb[2]

        d_qb = None
        if wbz == 0:
            d_qb = Vb
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
        return  next_q



