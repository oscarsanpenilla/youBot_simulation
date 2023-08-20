NewTask Description

Objective:
In this specific task, the primary challenge I undertook was to manipulate the mobile manipulator to pick up a block from the floor and subsequently place it on the robot's platform.

Implementations:
 - Joint Speed Limits: For the successful accomplishment of this task, it was imperative to introduce speed constraints on both the manipulator arm and the mobile base. These speed limitations were crucial in guiding the mobile manipulator's precise movements.
 - Arm: The speed of the manipulator arm was adjusted to ensure a smooth pick-and-place operation.
 - Robot's Base: To guarantee stability and precision during the trajectory, especially when handling the block, the speed of the robot's base was at times set to zero. This null speed ensures the base remains stationary during critical phases of the operation.

Noteworthy Observation:
Upon the successful placement of the block on the robot's platform, it might be evident that the block doesn't remain stable on the platform. This instability is attributed to the simulation not taking into account the collision meshes between the block and the platform. Thus, the block's observed behavior isn't a reflection of a software glitch or an error in the manipulator's operation but is an inherent characteristic of the current simulation settings.

controller type: feedforward + PI
kp = 2.9
ki = 0.01

start = np.array([
    [0.1699, 0, 0.9854, 0.22],
    [0, 1, 0, 0.0],
    [-0.9854, 0, 0.1699, 0.4713],
    [0, 0, 0, 1.0],
])

Tsc_init = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, -0.5],
    [0, 0, 1, 0.0],
    [0, 0, 0, 1.0],
])

Tsc_final = np.array([
    [-1, 0, 0, 0.2],
    [0, -1, 0, -0.5],
    [0, 0, 1, 0.175],
    [0, 0, 0, 1.0],
])