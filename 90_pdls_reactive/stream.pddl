(define (stream magpie-tamp)

  ;; Object Pose Stream ;;
  (:stream sample-object
    :inputs (?label)
    :domain (Graspable ?label)
    :outputs (?pose)
    :certified (Object ?label ?pose)
  )

  ;; Object Grasp Stream ;;
  (:stream sample-grasp
    :inputs (?label ?pose)
    :domain (Object ?label ?pose) ; We have to have a pose for this object before we can grasp it!
    :outputs (?effPose)
    :certified (Grasp ?pose ?effPose)
  )

  ;; IK Solver ;;
  (:stream inverse-kinematics
    :inputs (?effPose)
    :domain (Grasp ?pose ?effPose)
    :outputs (?config)
    :certified (and (IKSoln ?effPose ?config) (Conf ?config) )
  )

  ;; Free Placement Test ;;
  (:stream test-free-placment
    :inputs (?label ?pose)
    :domain (and (Object ?label ?objPose) (Grasp ?pose ?effPose))
    :certified (FreePlacement ?label ?pose)
  )

  ;; Safe Transit Test ;;
  (:stream test-safe-transit
    :inputs (?label ?bgnPose ?endPose)
    :domain (and (Object ?label ?objPose) (Grasp ?pose ?effPose))
    :certified (FreePlacement ?label ?pose)
  )

  ;; Safe Motion Test ;;
  (:stream test-safe-motion
    :inputs (?config1 ?config2)
    :domain (and (Conf ?config1) (Conf ?config2))
    :certified (FreePlacement ?label ?pose)
  )
)