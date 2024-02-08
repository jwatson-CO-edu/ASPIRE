(define (stream magpie-tamp)

  ;; Object Pose Stream ;;
  (:stream sample-object
    :inputs (?label)
    :domain (Graspable ?label)
    :outputs (?pose)
    :certified (and (WObject ?label ?pose) (Pose ?pose))
  )

  ;; Object Grasp Stream ;;
  (:stream sample-grasp
    :inputs (?label ?pose)
    :domain (and (Graspable ?label) (Pose ?pose)) ; ; We have to have a pose for this object before we can grasp it!
    :outputs (?effPose)
    :certified (and (EffPose ?effPose) (Grasp ?pose ?effPose))
  )

  ;; IK Solver ;;
  (:stream inverse-kinematics
    :inputs (?effPose)
    :domain (EffPose ?effPose)
    :outputs (?config)
    :certified (and (IKSoln ?effPose ?config) (Conf ?config) )
  )

  ;; Free Placement Test ;;
  (:stream test-free-placment
    :inputs (?label ?pose)
    :domain (and (Graspable ?label) (Pose ?pose))
    :certified (FreePlacement ?label ?pose)
  )

  ;; Safe Transit Test ;;
  (:stream test-safe-transit
    :inputs (?label ?bgnPose ?endPose)
    :domain (and (Graspable ?label) (Pose ?bgnPose) (Pose ?endPose))
    :certified (SafeTransit ?label ?bgnPose ?endPose)
  )

  ;; Safe Motion Test ;;
  (:stream test-safe-motion
    :inputs (?config1 ?config2)
    :domain (and (Conf ?config1) (Conf ?config2))
    :certified (SafeMotion ?config1 ?config2)
  )
)