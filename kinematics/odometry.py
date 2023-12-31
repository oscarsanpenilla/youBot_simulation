import numpy as np
from numpy import sin, cos
import modern_robotics as mr
from typing import List, Dict

class MobileBaseOdometry:
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

        # F - transformation matrix for calculating the base velocities
        self.F = r/4* np.array([
            [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
            [    1   ,   1    ,    1   ,   1     ],
            [   -1   ,   1    ,   -1   ,   1     ]
        ])

    @property
    def th_wheel_curr(self):
        # Property for accessing the current wheel angles
        return list(self._th_wheel_curr)

    @property
    def joint_max_speed(self):
        # Property for accessing the maximum joint speed
        return self._joint_max_speed

    @joint_max_speed.setter
    def joint_max_speed(self, value):
        # Setter for updating the maximum joint speed
        self._joint_max_speed = value

    def reset(self):
        # Resets the class to its initial configuration
        self.q_curr = self.q_init
        self._th_wheel_curr = np.zeros(4)
        self._total_time = 0.0

    def get_curr_base_trans(self):
        # Returns the current base transformation matrix
        phi = self.q_curr[0]
        skew = mr.VecToso3(np.array([0,0,1])*phi)
        rot = mr.MatrixExp3(skew)
        trans = np.array([self.q_curr[1], self.q_curr[2], 0.0963])
        T = np.eye(4)
        T[:3,:3] = rot
        T[:3, 3] = trans
        return T

    def update_base_config(self, u: List[float], timestep: float) -> List[float]:
        # Updates the base configuration based on the control input and timestep
        # u - control input (change in wheel angles)
        # timestep - time step

        self.q_curr = self.calc_new_base_config(u, timestep)
        self._th_wheel_curr += self._d_th
        return self.q_curr.copy()

    def calc_new_base_config(self, u: List[float], timestep: float) -> List[float]:
        # Calculates the new base configuration based on the control input and timestep
        # u - control input (change in wheel angles)
        # timestep - time step

        def is_almost_zero(x, epsilon=0.001):
            # Helper function to check if a value is almost zero
            return abs(x) < epsilon

        if len(u) != 4:
            raise RuntimeError("d_theta must be of size 4")

        # Enforce joint speed limits
        control_input = np.array([np.sign(speed) * min(abs(speed), self._joint_max_speed) for speed in u])

        phi, x, y = self.q_curr[0], self.q_curr[1], self.q_curr[2]

        self._d_th = control_input * timestep

        Vb = np.matmul(self.F, self._d_th)

        wbz = Vb[0]
        vbx = Vb[1]
        vby = Vb[2]

        d_qb = None
        if is_almost_zero(wbz):
            d_qb = np.array([0, vbx, vby])
        else:
            d_qb = np.array([
                [wbz],
                [(vbx*sin(wbz) + vby * (cos(wbz)-1))/wbz],
                [(vby*sin(wbz) + vbx * (1-cos(wbz)))/wbz],
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




