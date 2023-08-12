from unittest import TestCase
import numpy as np
from control.control import Control

np.set_printoptions(precision=3, suppress=True, linewidth=500)

class TestControl(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.base_init_config = [0, 0, 0]
        self.arm_init_config = [0, 0, 0.2, -1.6, 0]
        self.control = Control(self.base_init_config, self.arm_init_config)
    def test_feedback_control(self):
        dt = 0.01
        Xd = np.array([
            [ 0, 0, 1, 0.5],
            [ 0, 1, 0, 0  ],
            [-1, 0, 0, 0.5],
            [ 0, 0, 0, 1  ]
        ])
        Xd_next = np.array([
            [ 0, 0, 1, 0.6],
            [ 0, 1, 0, 0.0],
            [-1, 0, 0, 0.3],
            [ 0, 0, 0, 1]
        ])
        self.control.feedback_control(dt, Xd, Xd_next, kp=0, ki=0)

