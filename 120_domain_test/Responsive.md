# `DEV PLAN`
* `[>]` Domain Test
    - `[>]` Write Samplers
        * `[Y], sample-pose`: Sample open spots to place things, 2024-04-05: Written but UNTESTED
        * `[Y], sample-grasp`: Sample a grasp for the object (always the same for each pose), 2024-04-05: Written but UNTESTED
        * `[Y], inverse-kinematics`: Sample an IK sol'n for the object (always the same for each pose), 2024-04-05: Written but UNTESTED
        * `[>], plan-free-motion`: Get robot motion with nothing in the gripper
        * `[ ], plan-holding-motion`: Get robot motion with object in the gripper
        * `[ ], test-cfree-pose-pose`: Check that both poses are collision free
        * `[ ], test-cfree-approach-pose`: Test that the space above an object is collision free?
        * `[ ], test-cfree-traj-pose`: Test that the trajectory is collision free
    - `[ ]` Begin the problem with object locations
    - `[ ]` Simple problem test: 3 stacks of 2 blocks each
    