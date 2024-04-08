; https://planning.wiki/ref/pddl/problem
(define
    (problem theTwoTowers)
    (:domain pick-place-and-stack)
    (:objects ; ALL THE THINGS
        red
        ylw
        blu
        grn
        orn
        vio
        pose00
        pose01
        pose02
        pose03
        pose04
        pose05
        pose06
        pose07
        pose08
        pose09
        pose10
        pose11
    )
    (:init
        (Graspable red) ; Block Names
        (Graspable ylw)
        (Graspable blu)
        (Graspable grn)
        (Graspable orn)
        (Graspable vio)

        (Waypoint pose00) ; Beginning Poses
        (Waypoint pose01)
        (Waypoint pose02)
        (Waypoint pose03)
        (Waypoint pose04)
        (Waypoint pose05)

        (Waypoint pose06) ; Tower A Poses
        (Waypoint pose07)
        (Waypoint pose08)
        ; (FreePlacement pose06)
        ; (FreePlacement pose07)
        ; (FreePlacement pose08)
        (PoseAbove pose07 pose06)
        (PoseAbove pose08 pose07)

        (Waypoint pose09) ; Tower B Poses
        (Waypoint pose10)
        (Waypoint pose11)
        ; (FreePlacement pose09)
        ; (FreePlacement pose10)
        ; (FreePlacement pose11)
        (PoseAbove pose10 pose09)
        (PoseAbove pose11 pose10)

        (GraspObj red pose00) ; Beginning Object Configs
        (GraspObj ylw pose01)
        (GraspObj blu pose02)
        (GraspObj grn pose03)
        (GraspObj orn pose04)
        (GraspObj vio pose05)
    )
    (:goal
        (and
            (GraspObj red pose06) ; Tower A
            (Supported ylw red)
            (Supported blu ylw)

            (GraspObj grn pose09) ; Tower B
            (Supported orn grn)
            (Supported vio orn)

            (HandEmpty) ; Blocks free of "robot"
        )
    )
)