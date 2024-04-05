# `DEV PLAN`
* `[ ]` Domain Test
    - `[ ]` Write Samplers
        * `[ ], sample-pose`: Sample open spots to place things
        * `[ ], sample-grasp`: Sample a grasp for the object (always the same for each pose)
        * `[ ], inverse-kinematics`: Sample an IK sol'n for the object (always the same for each pose)
        * `[ ], plan-free-motion`: Get robot motion with nothing in the gripper
        * `[ ], plan-holding-motion`: Get robot motion with object in the gripper
        * `[ ], test-cfree-pose-pose`: Check that both poses are collision free
        * `[ ], test-cfree-approach-pose`: Test that the space above an object is collision free?
        * `[ ], test-cfree-traj-pose`: Test that the trajectory is collision free
    - `[ ]` Begin the problem with object locations
    - `[ ]` Simple problem test
    