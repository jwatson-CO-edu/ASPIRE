(define (stream magpie-tamp)

;;;;;;;;;; SYMBOL STREAMS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

  ;; Safe Transit Planner ;;
  (:stream path-planner
    :inputs (?label ?bgnPose ?endPose)
    :domain (and (Graspable ?label) (Pose ?bgnPose) (Pose ?endPose))
    :outputs (?traj)
    :certified (Path ?label ?bgnPose ?endPose ?traj)
  )

  ;; Stack Location Search ;;
  ; (:stream find-stack-place
  ;   :inputs (?labelDn1 ?labelDn2)
  ;   :domain (and (Graspable ?labelDn1) (Graspable ?labelDn2))
  ;   :outputs (?poseUp)
  ;   :certified (and (StackPlace ?poseUp ?labelDn1 ?labelDn2) (Pose ?poseUp))
  ; )
  (:stream find-stack-place
    :inputs (?labelDn1 ?labelDn2 ?poseDn1 ?poseDn2)
    :domain (and (Graspable ?labelDn1) (Graspable ?labelDn2) (Pose ?poseDn1) (Pose ?poseDn2))
    :outputs (?poseUp)
    :certified (and (Pose ?poseUp) (StackPlace ?labelDn1 ?labelDn2 ?poseUp ?poseDn1 ?poseDn2))
  )

  ;; Alternative Route Search ;;
  ; (:stream high-waypoint-sprinkler
  ;   :inputs (?poseDn)
  ;   :domain (Pose ?poseDn)
  ;   :outputs (?poseUp)
  ;   :certified (Pose ?poseUp)
  ; )

;;;;;;;;;; TESTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; Free Placement Test ;;
  ; (:stream test-free-placment
  ;   :inputs (?label ?pose)
  ;   :domain (and (Graspable ?label) (Pose ?pose))
  ;   :certified (FreePlacement ?label ?pose)
  ; )
  (:stream test-free-placment
    :inputs (?pose)
    :domain (and (Pose ?pose))
    :certified (FreePlacement ?pose)
  )

  ;; Safe Motion Test ;;
  (:stream test-safe-motion
    :inputs (?config1 ?config2)
    :domain (and (Conf ?config1) (Conf ?config2))
    :certified (SafeMotion ?config1 ?config2)
  )
)