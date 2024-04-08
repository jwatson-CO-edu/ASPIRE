; https://planning.wiki/ref/pddl/domain
(define (domain pick-place-and-stack)
  (:requirements :strips :negative-preconditions)

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;; Objects ;;;
    (GraspObj ?label ?pose) ; The concept of a named object at a pose
    (PoseAbove ?poseUp ?poseDn) ; The concept of a pose being supported by an object

    ;;; Domains ;;;
    (Graspable ?label); Name of a real object we can grasp
    (Waypoint ?pose) ; Model of any object we can go to in the world, real or not
  
    ;;; Conditions ;;;
    (Holding ?label) ; The label of the held object
    (HandEmpty) ; Is the robot hand empty?
    (Supported ?labelUp ?labelDn) ; Is the "up" object on top of the "down" object?
    (Blocked ?label)

    ;;; Checks ;;;
    ; (FreePlacement ?pose) ; Is there an open spot for placement?
  )

  ;;;;;;;;;; ACTIONS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    (:action pick
        :parameters (?label ?pose)
        :precondition (and 
                        ;; Knowns ;;
                        (HandEmpty)
                        ;; Requirements ;;
                        (Graspable ?label)
                        (Waypoint ?pose)
                        (not (Blocked ?label))
                        )
        :effect (and (Holding ?label) 
                     (not (HandEmpty))
                    ;  (FreePlacement ?pose)
                    ;  (increase (total-cost) 0.5)
                )
    )

    (:action move_holding
        :parameters (?label ?pose1 ?pose2)
        :precondition (and 
                        ;; Knowns ;;
                        (Holding ?label)
                        ;; Requirements ;;
                        (Waypoint ?pose1)
                        (Waypoint ?pose2)
                        (Graspable ?label)
                        (GraspObj ?label ?pose1)
                    )
        :effect (and (GraspObj ?label ?pose2)
                     (not (GraspObj ?label ?pose1))
                    ;  (increase (total-cost) 1)
                )
  )

    (:action place
        :parameters (?label ?pose)
        :precondition (and 
                        ;; Knowns ;;
                        (Holding ?label)
                        ;; Requirements ;;
                        (Waypoint ?pose)
                        (Graspable ?label)
                        (GraspObj ?label ?pose)
                        )
        :effect (and (HandEmpty) 
                     (not (Holding ?label)) 
                    ;  (not (FreePlacement ?pose))
                    ;  (increase (total-cost) 0.5)
                )
    )

    (:action stack
        :parameters (?labelUp ?labelDn ?poseUp ?poseDn)
        :precondition (and 
                        ;; Knowns ;;
                        (Holding ?labelUp)
                        ;; Requirements ;;
                        (Waypoint ?poseUp)
                        (Waypoint ?poseDn)
                        (Graspable ?labelUp)
                        (Graspable ?labelDn)
                        ; (FreePlacement ?poseUp)
                        (GraspObj ?labelDn ?poseDn)
                        (GraspObj ?labelUp ?poseUp)
                        (PoseAbove ?poseUp ?poseDn)
                        )
        :effect (and (HandEmpty) 
                     (not (Holding ?labelUp))
                     (Supported ?labelUp ?labelDn)
                    ;  (not (FreePlacement ?poseUp)) ; Planner does dumb things if this isn't here
                     (Blocked ?labelDn)
                    ;  (increase (total-cost) 0)
                )
    )
)