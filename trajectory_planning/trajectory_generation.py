import numpy as np
from scipy.linalg import logm, expm
from typing import List, Dict
import modern_robotics as mr

class Trajectory:
    def __init__(self):
        self._k = 1
        self._num_trans_per_second = 0.01
        self._Xstart: np.array = None
        self._Xend: np.array = None
        self._X: np.array = None

    def generate(self, traj_duration: float):
        if self._Xstart is None:
            raise RuntimeError("Xstart is unset, use set_start_trans method")
        if self._Xend is None:
            raise RuntimeError("Xend is unset, use set_start_trans method")

        T = traj_duration
        screw_path = []
        mat_log = mr.MatrixLog6(np.matmul(np.linalg.inv(self._Xstart), self._Xend))
        t_list = np.arange(0, traj_duration + 0.01, self._num_trans_per_second)
        s_list =  [3/(T**2)*(t**2) - 2/(T**3)*(t**3) for t in t_list]
        for s in s_list:
            exp_mat = expm(s * mat_log)
            X_s = np.matmul(self._Xstart, exp_mat)
            rot_mat = X_s[:3, :3]
            trans_mat = X_s[:3, 3]
            screw_path.append(rot_mat.flatten().tolist() + trans_mat.flatten().tolist())

        return np.array(screw_path).tolist()



    def set_start_trans(self, trans):
        self._Xstart = np.array(trans)

    def set_end_trans(self, trans):
        self._Xend = np.array(trans)
