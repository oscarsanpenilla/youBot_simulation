import numpy as np
import modern_robotics as mr
from kinematics.forward_kin import ForwardKin
from kinematics.mobile_manipulator_kin import MobileManipulatorKin
from kinematics.odometry import FourWheeledMecanumOdometry


class Control:
    def __init__(self, robot_kin: MobileManipulatorKin):
        """
        Initialize the Control class.

        Args:
            robot_kin (MobileManipulatorKin): Object representing the robot's kinematics.
        """
        self._robot_kin = robot_kin
        self._sum_Xerr = np.zeros(6)  # Initialize the cumulative error vector

    def feedback_control(self, timestep: float, Xd, Xd_next, kp, ki, print_enabled=False):
        """
        Perform feedback control to generate the control output.

        Args:
            timestep (float): Duration of the time step.
            Xd: Desired end effector transformation (desired pose).
            Xd_next: Next desired end effector transformation (next desired pose).
            kp: Proportional gain matrix.
            ki: Integral gain matrix.
            print_enabled (bool): Whether to print intermediate values for debugging purposes.

        Returns:
            np.ndarray: Control output (joint velocities or forces).

        """
        dt = timestep
        X = self._robot_kin.get_end_effector_transform()  # Current end effector transformation
        X_inv = np.linalg.inv(X)

        Xd_inv = np.linalg.inv(Xd)
        skew_Xerr = mr.MatrixLog6(np.matmul(X_inv, Xd))  # Calculate the skew-symmetric matrix of the difference between X and Xd
        self.Xerr = mr.se3ToVec(skew_Xerr)
        skew_twist = 1/dt * mr.MatrixLog6(np.matmul(Xd_inv, Xd_next))
        twist_b = mr.se3ToVec(skew_twist)

        # Calculate Jacobian
        Je = self._robot_kin.get_curr_jacobian()
        pseudoJ = np.linalg.pinv(Je)

        # Control law
        self._sum_Xerr += self.Xerr * dt  # Accumulate the error over time
        adjoint_Xinv_Xd = mr.Adjoint(np.matmul(X_inv, Xd))
        feedforward_control = np.matmul(adjoint_Xinv_Xd, twist_b)
        proportional_control = np.matmul(kp*np.eye(len(self.Xerr)), self.Xerr)
        integral_control = np.matmul(ki*np.eye(len(self.Xerr)), self._sum_Xerr)

        twist = feedforward_control + proportional_control + integral_control  # Calculate the overall twist

        ctrl_output = np.matmul(pseudoJ, twist)  # Calculate the control output by multiplying the pseudo-inverse Jacobian with the twist

        if print_enabled:
            print(f"X:\n {X}")
            print(f"Xerr:\n {self.Xerr}")
            print(f"feedforward_control:\n {feedforward_control}")
            print(f"twist_b:\n {twist_b}")
            print(f"Je:\n {Je}")
            print(f"pseudoJ:\n {pseudoJ}")
            print(f"control_output:\n {ctrl_output}")

        return ctrl_output

    def get_curr_error(self):
        """
        Get the current error vector.

        Returns:
            np.ndarray: Current error vector.
        """
        return self.Xerr


