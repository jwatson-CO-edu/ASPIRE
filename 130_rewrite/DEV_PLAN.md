# `DEV PLAN`
* `[ ]` Implement new domain "11"
    - `[Y]` Copy "11" to main, 2024-04-25: Confirmed to unstack-swap-restack at the online static planner
    - `[>]` `table` must exist as a `Base`
    - `[ ]` **Every** pose **must** have a `PoseAbove`
    - `[ ]` Supported detector must also instantiate `Blocked`
        * `[ ]` Try using a derived predicate
        * `[ ]` Otherwise, just ground it like everything else at Phase 2
    - `[ ]` Injected swap poses musn't interfere with each other
* `[ ]` Check that similar poses do not shadow each other w/ different names