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

  ; (:stream sample-arch
  ;   :inputs (?objUp ?objDn1 ?objDn2)
  ;   :domain (and (Stackable ?objUp ?objDn1) (Stackable ?objUp ?objDn2))
  ;   :outputs (?p)
  ;   :certified (and (Pose ?objUp ?p) (Supported ?objUp ?p ?objDn1) (Supported ?objUp ?p ?objDn2) (Arched ?objUp ?objdn1 ?objdn2) )
  ; )

  

  ; ;; MP Path Between Configs ;;
  ; (:stream plan-free-motion
  ;   :inputs (?q1 ?q2)
  ;   :domain (and (Conf ?q1) (Conf ?q2))
  ;   :outputs (?t)
  ;   :certified (FreeMotion ?q1 ?t ?q2)
  ; )

  

  ; ;; Pose Collision Test ;;
  ; (:stream test-cfree-pose-pose
  ;   :inputs (?o1 ?p1 ?o2 ?p2)
  ;   :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
  ;   :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  ; )

  ;; Trajectory Collision Test ;;
  ; (:stream test-cfree-pose-pose
  ;   :inputs (?o1 ?p1 ?o2 ?p2)
  ;   :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
  ;   :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  ; )
)