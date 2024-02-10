(define (stream magpie-tamp)

;;;;;;;;;; SYMBOL STREAMS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ; ;; Stack Location Search ;;
  (:stream find-stack-place
    :inputs (?labelUp ?poseDn1 ?poseDn2)
    :domain (and (Graspable ?labelUp) (Pose ?poseDn1) (Pose ?poseDn2))
    :outputs (?poseUp)
    :certified (and (StackPlace ?labelUp ?poseUp ?poseDn1 ?poseDn2))
  )

    ;; Object Grasp Stream ;;
  (:stream sample-grasp
    :inputs (?label ?pose)
    :domain (and (Graspable ?label) (Pose ?pose)) ; We have to have a pose for this object before we can grasp it!
    :outputs (?effPose)
    :certified (and (EffPose ?effPose) (Grasp ?label ?pose ?effPose))
  )

  ; Object Pose Stream ;;
  (:stream sample-object
    :inputs (?label)
    :domain (Graspable ?label)
    :outputs (?pose)
    :certified (and (Obj ?label ?pose) (Pose ?pose))
  )

  ;; IK Solver ;;
  (:stream inverse-kinematics
    :inputs (?effPose)
    :domain (EffPose ?effPose)
    :outputs (?config)
    :certified (and (IKSoln ?effPose ?config) (Conf ?config) )
  )

  ;; Safe Transit Planner ;;
  (:stream path-planner
    :inputs (?label ?bgnPose ?endPose)
    :domain (and (Graspable ?label) (Pose ?bgnPose) (Pose ?endPose))
    :outputs (?traj)
    :certified (Path ?label ?bgnPose ?endPose ?traj)
  )


;;;;;;;;;; TESTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; Safe Motion Test ;;
  (:stream test-safe-motion
    :inputs (?config1 ?config2)
    :domain (and (Conf ?config1) (Conf ?config2))
    :certified (SafeMotion ?config1 ?config2)
  )

  ; Free Placement Search ;;
  (:stream test-free-placment
    :inputs (?label ?pose)
    :domain (and (Graspable ?label) (Pose ?pose))
    :certified (and (FreePlacement ?label ?pose))
  )
)