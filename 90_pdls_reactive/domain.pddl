(define (domain pick-and-place)
  (:requirements :strips )

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;; Symbols ;;;
    (WObject ?label ?pose); Sample from world -or- Produced by actions
    (IKSoln ?pose ?config) ; Sample from pose
    (Grasp ?pose ?effPose) ; Sample from pose

    ;;; Domains ;;;
    (Graspable ?label) ; Used by "stream.pddl"
    (Conf ?config) ; Used by "stream.pddl"
    (Pose ?pose)
    (EffPose ?pose)
  
    ;;; States ;;;
    (Holding ?label) ; From Pick
    (HandEmpty) ; From Place
    (AtConf ?config) ; From moves
    (AtPose ?effPose) ; From Move Holding

    ;;; Checks ;;;
    (FreePlacement ?label ?pose) ; Checked by world
    (SafeTransit ?label ?bgnPose ?endPose ) ; Checked by world
    (SafeMotion ?config1 ?config2) ; Checked by world
  )

  ;;;;;;;;;; ACTIONS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:action move_q
    :parameters (?config1 ?config2)
    :precondition (and (SafeMotion ?config1 ?config2)
                       (AtConf ?config1) 
                  )
    :effect (and (AtConf ?config2)
                 (not (AtConf ?config1)) ) 
  )

  (:action move_free
    :parameters (?effPose1 ?effPose2 ?config1 ?config2)
    :precondition (and (AtPose ?effPose1)
                       (HandEmpty)
                       (IKSoln ?effPose2 ?config2) 
                       (IKSoln ?effPose1 ?config1)
                  )
    :effect (and (AtPose ?effPose2)
                 (not (AtPose ?effPose1))
                 (AtConf ?config2)
                 (not (AtConf ?config1))
            ) 
  )

  (:action pick
    :parameters (?label ?pose ?effPose)
    :precondition (and (WObject ?label ?pose)
                       (Grasp ?pose ?effPose)
                       (AtPose ?effPose)
                       (HandEmpty)
                  )
    :effect (and (Holding ?label) 
                 (not (HandEmpty)))
  )

  (:action move_holding
    :parameters (?label ?bgnPose ?endPose ?effPose1 ?effPose2 ?config1 ?config2)
    :precondition (and (Grasp ?bgnPose ?effPose1)
                       (Grasp ?endPose ?effPose2)
                       (Holding ?label)
                       (IKSoln ?effPose2 ?config2) 
                       (IKSoln ?effPose1 ?config1)
                       (AtPose ?effPose1)
                  )
    :effect (and (WObject ?label ?endPose) 
                 (not (WObject ?label ?bgnPose))
                 (AtPose ?effPose2)
                 (not (AtPose ?effPose1))
                 (AtConf ?config2)
                 (not (AtConf ?config1))
            )
  )

  (:action place
    :parameters (?label ?pose ?effPose)
    :precondition (and (FreePlacement ?label ?pose)
                       (Holding ?label)
                       (Grasp ?pose ?effPose)
                       (AtPose ?effPose)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?label)))
  )  
)