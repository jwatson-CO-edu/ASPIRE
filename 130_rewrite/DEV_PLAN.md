# `DEV PLAN`
* `[ ]` Implement new domain "11"
    - `[Y]` Copy "11" to main, 2024-04-25: Confirmed to unstack-swap-restack at the online static planner
    - `[Y]` `table` must exist as a `Base`, 2024-04-25: Added to `phase_2_Conditions`
    - `[Y]` **Every** pose **must** have a `PoseAbove`
        * 2024-04-25: For stacking, this is provided by the `sample-above` stream
        * 2024-04-25: For placement, this is provided by:
            - `BaselineTAMP.ground_relevant_predicates_noisy`
    - `[Y]` `Supported` detector must also instantiate `Blocked`, 2024-04-25: It already does
    - `[Y]` Injected swap poses musn't interfere with each other, 2024-04-25: Adds new poses to verfication each iteration
* `[>]` Check that similar poses do not shadow each other w/ different names
    - `[>]` Check against the swap version
* `[ ]` Use only neccessary streams, Check correctness