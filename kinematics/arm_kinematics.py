from typing import List

import numpy as np
import modern_robotics as mr

class ArmKinematics:
    def __init__(self, init_q: List[float]):
        """
        Initializes the ForwardKin class with initial joint angles.

        Args:
            init_q (List[float]): A list of initial joint angles.
        """
        self._init_q = np.array(init_q)
        self._curr_arm_th = self._init_q.copy()
        self._joint_max_speed = 9e9
        self._joint_speeds = np.array([self._joint_max_speed] * len(self._init_q))

        # Transformation matrix from the last joint frame to the end effector frame
        self.M0e = np.array([
            [1, 0, 0, 0.0330],
            [0, 1, 0, 0.0   ],
            [0, 0, 1, 0.6546],
            [0, 0, 0, 1.0   ],
        ])

        # List of joint screw axes
        self.B_list = np.array([
            [0,  0, 1,  0.    , 0.033, 0],
            [0, -1, 0, -0.5076, 0    , 0],
            [0, -1, 0, -0.3526, 0    , 0],
            [0, -1, 0, -0.2176, 0    , 0],
            [0,  0, 1,  0.    , 0    , 0],
        ]).T

    @property
    def joint_max_speed(self):
        """
        Gets the maximum joint speed.

        Returns:
            float: Maximum joint speed.
        """
        return self._joint_max_speed

    @joint_max_speed.setter
    def joint_max_speed(self, value):
        """
        Sets the maximum joint speed.

        Args:
            value (float): Maximum joint speed.
        """
        self._joint_max_speed = abs(value)

    def get_new_arm_config(self, joint_speeds: List[float], timestep: float) -> List[float]:
        """
        Calculates the new arm configuration based on the current joint speeds and timestep.

        Args:
            joint_speeds (List[float]): List of joint speeds.
            timestep (float): Time step.

        Returns:
            List[float]: New arm configuration (joint angles).
        """
        # Enforce joint speed limits
        j_speeds = np.array([np.sign(speed) * min(abs(speed), self._joint_max_speed) for speed in joint_speeds])

        new_angles = self._curr_arm_th + j_speeds * timestep
        return new_angles.tolist()

    def update_arm_config(self, joint_speeds: List[float], timestep: float) -> List[float]:
        """
        Updates the current arm configuration based on the joint speeds and timestep.

        Args:
            joint_speeds (List[float]): List of joint speeds.
            timestep (float): Time step.

        Returns:
            List[float]: Updated arm configuration (joint angles).
        """
        self._curr_arm_th = self.get_new_arm_config(joint_speeds, timestep)
        return self._curr_arm_th.copy()

    def forward_kinematics(self, theta_list: List[float]):
        """
        Calculates the end effector transformation matrix using forward kinematics.

        Args:
            theta_list (List[float]): List of joint angles.

        Returns:
            np.ndarray: Transformation matrix from the base frame to the end effector frame.
        """
        Te0 = mr.FKinBody(self.M0e, self.B_list, np.array(theta_list))
        return Te0

    def inverse_kinematics(self, transform, seed=None):
        """
        Calculates the inverse kinematics solution for a given end effector transformation matrix.

        Args:
            transform (np.ndarray): End effector transformation matrix.
            seed (optional): Initial guess for joint angles. Default to the current arm configuration.

        Returns:
            Tuple[List[float], bool]: Tuple containing the joint angles solution and a boolean indicating if a solution was found.
        """
        if seed is None:
            seed = self._curr_arm_th.copy()

        return mr.IKinBody(self.B_list, self.M0e, transform, seed, 0.01, 0.001)

    def set_from_IK(self, transform, seed=None):
        """
        Sets the current arm configuration using inverse kinematics to reach a desired end effector transformation matrix.

        Args:
            transform (np.ndarray): Desired end effector transformation matrix.
            seed (optional): Initial guess for joint angles. Default to the current arm configuration.

        Raises:
            RuntimeError: If no inverse kinematics solution is found.
        """
        arm_q, sol_found = self.inverse_kinematics(transform, seed)
        if sol_found:
            self._curr_arm_th = list(arm_q)
        else:
            raise RuntimeError("Unable to set from Inverse Kinematics")

    def body_jacobian(self, theta_list: List[float]):
        """
        Calculates the body Jacobian matrix for a given set of joint angles.

        Args:
            theta_list (List[float]): List of joint angles.

        Returns:
            np.ndarray: Body Jacobian matrix.
        """
        bodyJ = mr.JacobianBody(self.B_list, np.array(theta_list))
        return bodyJ

    def get_curr_body_jacobian(self):
        """
        Calculates the current body Jacobian matrix based on the current arm configuration.

        Returns:
            np.ndarray: Current body Jacobian matrix.
        """
        return self.body_jacobian(self._curr_arm_th)

    def get_curr_config(self):
        """
        Gets the current arm configuration (joint angles).

        Returns:
            List[float]: Current arm configuration (joint angles).
        """
        return self._curr_arm_th.copy()

    def get_curr_endeffector_transform(self):
        """
        Calculates the current end effector transformation matrix based on the current arm configuration.

        Returns:
            np.ndarray: Current end effector transformation matrix.
        """
        return self.forward_kinematics(self._curr_arm_th)

