import numpy as np
import modern_robotics as mr
from kinematics.forward_kin import ForwardKin
from kinematics.odometry import FourWheeledMecanumOdometry


class Control:
    def __init__(self, base_init_config, arm_init_config):
        l = 0.47 / 2
        w = 0.3 / 2
        r = 0.0475
        self._base_init_config = base_init_config
        self._arm_init_config = arm_init_config
        self.base_odom = FourWheeledMecanumOdometry(r, w, l, base_init_config)
        self.arm_kin = ForwardKin(arm_init_config)
        self._sum_Xerr = np.zeros(6)

    def feedback_control(self, timestep: float, Xd, Xd_next, kp, ki):
        Tb0 = np.array([
            [1, 0, 0, 0.1662],
            [0, 1, 0, 0.    ],
            [0, 0, 1, 0.0026],
            [0, 0, 0, 1.    ],
        ])
        T0b = np.linalg.inv(Tb0)

        dt = timestep
        wheel_speeds = [0]*4
        joint_speeds = [0]*5
        q_base = self.base_odom.update_base_config(wheel_speeds, dt)
        Tsb = self.base_odom.get_curr_base_trans()
        q_arm = self.arm_kin.update_arm_config(joint_speeds, dt)
        T0e = self.arm_kin.forward_kinematics(q_arm)
        Te0 = np.linalg.inv(T0e)
        Ts0 = np.matmul(Tsb, Tb0)
        X = np.matmul(Ts0, T0e)
        X_inv = np.linalg.inv(X)

        Xd_inv = np.linalg.inv(Xd)
        skew_Xerr = mr.MatrixLog6(np.matmul(X_inv, Xd))
        Xerr = mr.se3ToVec(skew_Xerr)
        skew_twist = 1/dt * mr.MatrixLog6(np.matmul(Xd_inv, Xd_next))
        twist_b = mr.se3ToVec(skew_twist)

        # Calc Jacobian
        r, l, w = self.base_odom.r, self.base_odom.l, self.base_odom.w
        F = r/4 * np.array([
            [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w) ],
            [   1    ,     1  ,    1   ,    1     ],
            [  -1    ,     1  ,   -1   ,    1     ],
        ])
        F6 = np.zeros((6, 4))
        F6[2:5,:] = F

        Jbase = np.matmul(mr.Adjoint(np.matmul(Te0, T0b)), F6)
        Jarm  = self.arm_kin.body_jacobian(q_arm)
        Je = np.hstack([Jbase, Jarm])
        pseudoJ = np.linalg.pinv(Je)
        adjoint_Xinv_Xd = mr.Adjoint(np.matmul(X_inv, Xd))
        feedforward_control = np.matmul(adjoint_Xinv_Xd, twist_b)
        proportional_control = np.matmul(kp*np.eye(len(Xerr)), Xerr)

        self._sum_Xerr += Xerr * dt
        integral_control = np.matmul(ki*np.eye(len(Xerr)), self._sum_Xerr)

        twist = feedforward_control + proportional_control + integral_control
        control_ouput = np.matmul(pseudoJ, twist)

        print (f"X:\n {X}")
        print (f"Xerr:\n {Xerr}")
        print (f"feedforward_control:\n {feedforward_control}")
        print (f"twist_b:\n {twist_b}")
        print (f"Je:\n {Je}")
        print (f"pseudoJ:\n {pseudoJ}")
        print (f"control_ouput:\n {control_ouput}")
