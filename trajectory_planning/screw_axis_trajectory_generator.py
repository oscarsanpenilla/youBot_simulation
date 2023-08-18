import numpy as np
from scipy.linalg import expm
import modern_robotics as mr


class ScrewAxisTrajectoryGenerator:
    def __init__(self):
        self._num_trans_per_second = 0.01
        self._Xstart: np.array = None  # Stores the start pose of the trajectory
        self._Xend: np.array = None  # Stores the end pose of the trajectory
        self._X: np.array = None  # Stores the current pose during trajectory generation

    def generate(self, traj_duration: float) -> np.array:
        """
        Generates a screw axis trajectory given the trajectory duration.

        Parameters:
            traj_duration (float): The duration of the desired trajectory.

        Returns:
            np.array: The generated screw axis trajectory.

        Raises:
            RuntimeError: If the start or end pose is not set.
        """
        if self._Xstart is None:
            raise RuntimeError("Xstart is unset, use set_start_pose method")
        if self._Xend is None:
            raise RuntimeError("Xend is unset, use set_end_pose method")

        T = traj_duration
        screw_path = []  # Stores the generated trajectory
        mat_log = mr.MatrixLog6(np.matmul(np.linalg.inv(self._Xstart), self._Xend))  # Calculates the matrix logarithm
        t_list = np.arange(0, traj_duration + 0.01, self._num_trans_per_second)  # Generates points in time
        s_list = [3 / (T ** 2) * (t ** 2) - 2 / (T ** 3) * (t ** 3) for t in t_list]  # Generates s values for interpolation
        for s in s_list:
            exp_mat = expm(s * mat_log)  # Exponentiates the matrix logarithm
            X_s = np.matmul(self._Xstart, exp_mat)  # Calculates the pose at given s
            screw_path.append(X_s)

        return np.array(screw_path)

    def generate_trajectories(self, start, goals, times, gripper_states) -> np.array:
        """
        Generates multiple screw axis trajectories given multiple goals, times, and gripper states.

        Parameters:
            start: The start pose of the trajectory.
            goals: A list of end poses for each trajectory.
            times: A list of durations for each trajectory.
            gripper_states: A list of gripper states for each trajectory.

        Returns:
            np.array: An array of generated screw axis trajectories.
        """
        trajectories = []
        self.set_start_pose(start)
        for goal, time, gripper_state in zip(goals, times, gripper_states):
            self.set_end_pose(goal)
            traj = self.generate(time)
            trajectories.append(traj)
            self.set_start_pose(goal)
        trajectories = np.array(trajectories)
        return trajectories

    def set_start_pose(self, trans):
        """
        Sets the start pose of the trajectory.

        Parameters:
            trans: The start pose as a transformation matrix.
        """
        self._Xstart = np.array(trans).copy()

    def set_end_pose(self, trans):
        """
        Sets the end pose of the trajectory.

        Parameters:
            trans: The end pose as a transformation matrix.
        """
        self._Xend = np.array(trans).copy()


