import numpy as np
from numpy import sin, cos
import sympy as sp
from typing import List, Dict

class FourWheeledMecanumOdometry:
    def __init__(self, r, w, l):
        self.r = r
        self.w = w
        self.l = l

    def get_new_base_config(self, d_theta: List[float]) -> List[float]:

        if len(d_theta) != 5:
            raise RuntimeError("d_theta must be of size 5")

        l = self.l
        r = self.r
        w = self.w
        phi = d_theta[0]
        q_init = d_theta[1:]

        F = r/4* np.array([
            [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
            [    1   ,   1    ,    1   ,   1     ],
            [   -1   ,   1    ,   -1   ,   1     ]
        ])

        th = np.array(d_theta).T
        Vb = np.matmul(F, th)

        vbx = Vb[0]
        vby = Vb[1]
        wbz = Vb[2]

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
            [0, sin(phi),  cos(phi)],
        ])
        d_q = np.matmul(chassis_rot, d_qb)

        next_q = q_init + d_q
        return  next_q



