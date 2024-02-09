(define (stream magpie-tamp)

;;;;;;;;;; SYMBOL STREAMS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; Stack Location Search ;;
  (:stream find-stack-place
    :inputs (?labelUp ?tgtDn1 ?tgtDn2)
    :domain (and (Graspable ?labelUp) (Tgt ?tgtDn1) (Tgt ?tgtDn2))
    :outputs (?poseUp)
    :certified (and (Pose ?poseUp) (StackPlace ?labelDn1 ?labelDn2 ?poseUp ?poseDn1 ?poseDn2))
  )

  ;; Free Placement Search ;;
  (:stream find-free-placment
    :inputs (?label ?pose)
    :domain (and (Graspable ?label) (Pose ?pose))
    :outputs (?tgt)
    :certified (and (Tgt ?tgt) (FreePlacement ?tgt))
  )

  ;; Object Pose Stream ;;
  (:stream sample-object
    :inputs (?label)
    :domain (Graspable ?label)
    :outputs (?obj)
    :certified (Obj ?obj)
  )

  ;; Object Grasp Stream ;;
  (:stream sample-grasp
    :inputs (?effPose)
    :domain (and (Obj ?obj)) ; ; We have to have a pose for this object before we can grasp it!
    :outputs (?effPose)
    :certified (and (Grasp ?label ?pose ?effPose))
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

  

  ;; Alternative Route Search ;;
  ; (:stream high-waypoint-sprinkler
  ;   :inputs (?poseDn)
  ;   :domain (Pose ?poseDn)
  ;   :outputs (?poseUp)
  ;   :certified (Pose ?poseUp)
  ; )

;;;;;;;;;; TESTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  

  ;; Safe Motion Test ;;
  (:stream test-safe-motion
    :inputs (?config1 ?config2)
    :domain (and (Conf ?config1) (Conf ?config2))
    :certified (SafeMotion ?config1 ?config2)
  )
)