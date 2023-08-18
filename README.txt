# Capstone Project: Mobile Manipulator Control Software

## Overview:
For my capstone project, I have developed a software system that controls a mobile manipulator. The primary objective is to enable the manipulator to seamlessly pick up a block from its initial position and relocate it to a specified goal configuration. The culmination of my efforts is visualized through an animation illustrating one feasible solution.

## Developed Components:
 - Trajectory Planner: This component formulates the ideal trajectory for the end-effector, ensuring a precise path of movement.

 - Feedback Controller: I implemented this to correct any initial discrepancies and disturbances. This ensures that the end-effector consistently follows the trajectory outlined by the planner.

 - Simulation: Within this, I've integrated odometry to deduce the movement of the chassis, basing this on the wheel velocities commanded by the feedback controller. I've designed it so the feedback controller can readily access these odometry predictions.

## Trajectory Segments:
The entire operation of block manipulation has been divided into eight distinct segments:

- Segment 1: The gripper approaches a standoff position right above the block. Initially, it starts off the predetermined path, but my feedback controller efficiently brings it back to the reference trajectory.

- Segment 2: The gripper moves downwards, aligning itself with the block.

- Segment 3: The gripper closes, ensuring a firm grasp on the block.

- Segment 4: Securely holding the block, the gripper lifts it.

- Segment 5: The gripper smoothly moves the block, positioning it above its intended final spot.

- Segment 6: The gripper carefully places the block down.

- Segment 7: After ensuring the block is stable, the gripper releases it.

- Segment 8: Finally, the gripper withdraws, completing the operation.

## Some Details:
 - The generated trajectories are primarily in the task space. However, they are convertible to the joint space.
 - I employed a 3rd order polynomial to transform the paths into trajectories.
 - The depicted trajectories were produced leveraging the concept of screw trajectories.
 - The software provides an option to assign speed limits to both the robot chassis and the arm. This feature was crucial in preventing the mobile base from moving during the "NewTask".

## Comments:
 - After integrating the PI + feedforward controller, it was challenging to determine a well-tuned set of ki and kp matrix constants yielding satisfactory performance. The transient response was evident, yet the trajectory (particularly segment 1) should be smooth, with minimal error and no overshoot.
 - Concerning the overshoot scenario, identifying an appropriate set showcasing overshoot and minor oscillation was equally challenging. However, errors ought to be rectified before the culmination of trajectory segment 1.
 - Debugging was necessary when testing the pure feedforward controller.
 - There's potential to refactor the code, enhancing reusability and clarity.

## Output:
Upon executing the main.py script, a simulation can be run using the generated CoppeliaSim.csv file and the provided Scene6_youBot_cube. Which showcase the motion of the mobile manipulator, all governed by the Feedback + PI controller I implemented.

----

# NewTask Description

## Objective:
In this specific task, the primary challenge I undertook was to manipulate the mobile manipulator to pick up a block from the floor and subsequently place it on the robot's platform.

## Implementations:
 - Joint Speed Limits: For the successful accomplishment of this task, it was imperative to introduce speed constraints on both the manipulator arm and the mobile base. These speed limitations were crucial in guiding the mobile manipulator's precise movements.
 - Arm: The speed of the manipulator arm was adjusted to ensure a smooth pick-and-place operation.
 - Robot's Base: To guarantee stability and precision during the trajectory, especially when handling the block, the speed of the robot's base was at times set to zero. This null speed ensures the base remains stationary during critical phases of the operation.

### Observation:
Upon the successful placement of the block on the robot's platform, it might be evident that the block doesn't remain stable on the platform. This instability is attributed to the simulation not taking into account the collision meshes between the block and the platform. Thus, the block's observed behavior isn't a reflection of a software glitch or an error in the manipulator's operation but is an inherent characteristic of the current simulation settings.


