(define (stream magpie-tamp)

  ;; Object Pose Stream ;;
  (:stream sample-pose
    :inputs (?objUp ?objDn)
    :domain (Stackable ?objUp ?objDn)
    :outputs (?p)
    :certified (and (Pose ?objUp ?p) (Supported ?objUp ?p ?objDn))
  )

  (:stream sample-arch
    :inputs (?objUp ?objDn1 ?objDn2)
    :domain (and (Stackable ?objUp ?objDn1) (Stackable ?objUp ?objDn2))
    :outputs (?p)
    :certified (and (Pose ?objUp ?p) (Supported ?objUp ?p ?objDn1) (Supported ?objUp ?p ?objDn2) (Arched ?objUp ?objdn1 ?objdn2) )
  )

  ;; Object Grasp Stream ;;
  (:stream sample-grasp
    :inputs (?o ?p)
    :domain (and (Pose ?o ?p)) ; We have to have a pose for this object before we can grasp it!
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  ;; MP Path Between Configs ;;
  (:stream plan-free-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :outputs (?t)
    :certified (FreeMotion ?q1 ?t ?q2)
  )

  ;; IK Solver ;;
  (:stream inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q ?t)
    :certified (and (Conf ?q) (Traj ?t) (Kin ?o ?p ?g ?q ?t) 
                    (CFreeTrajPose ?t ?o ?p) ) ; WARNING: This one is NEW
  )

  ;; Pose Collision Test ;;
  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  )

  ;; Trajectory Collision Test ;;
  ; (:stream test-cfree-pose-pose
  ;   :inputs (?o1 ?p1 ?o2 ?p2)
  ;   :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
  ;   :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  ; )
)