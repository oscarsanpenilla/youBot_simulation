from unittest import TestCase
from kinematics.odometry import FourWheeledMecanumOdometry


class TestFourWheeledMecanumOdometry(TestCase):

    # def __init__(self):
    #     super().__init__()

    def setUp(self) -> None:
        super().setUp()
        l = 0.47 / 2
        w = 0.3 / 2
        r = 0.0475
        self.robot = FourWheeledMecanumOdometry(r, w, l)

    def test_get_new_base_config(self):
        next_q = self.robot.get_new_base_config([0, 0, 0, 0, 0])
        print(next_q)
