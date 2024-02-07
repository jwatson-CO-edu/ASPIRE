(define (domain pick-and-place)
  (:requirements :strips :equality 
                 :negative-preconditions :derived-predicates )

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;; Symbols ;;;
    (Object ?label ?pose); Sample from world -or- Produced by actions
    (IKSoln ?pose ?config) ; Sample from pose
    (Grasp ?pose ?effPose) ; Sample from pose

    ;;; Domains ;;;
    (Graspable ?label) ; Used by "stream.pddl"
    (Conf ?config) ; Used by "stream.pddl"
  
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

  (:action move_holding
    :parameters (?config1 ?config2 ?label ?bgnPose ?endPose ?effPose1 ?effPose2)
    :precondition (and (Holding ?label)
                       (AtConf ?config1) 
                       (Grasp ?bgnPose ?effPose1)
                       (Grasp ?endPose ?effPose2)
                       (IKSoln ?effPose2 ?config2)
                       (Object ?label ?bgnPose)
                       (SafeMotion ?config1 ?config2)
                       (SafeTransit ?label ?bgnPose ?endPose )
                  )
    :effect (and (AtPose ?effPose2)
                 (not (AtPose ?effPose1))
                 (AtConf ?config2)
                 (not (AtConf ?config1))
                 (Object ?label ?endPose) 
                 (not (Object ?label ?bgnPose))
            )
  )

  (:action pick
    :parameters (?label ?pose ?effPose)
    :precondition (and (Object ?label ?pose)
                       (Grasp ?pose ?effPose)
                       (AtPose ?effPose)
                       (HandEmpty)
                  )
    :effect (and (Holding ?label) 
                 (not (HandEmpty)))
  )

  (:action move_free
    :parameters (?config1 ?config2 ?effPose1 ?effPose2)
    :precondition (and (SafeMotion ?config1 ?config2)
                       (AtConf ?config1) 
                       (IKSoln ?effPose2 ?config2)
                       (HandEmpty) 
                  )
    :effect (and (AtConf ?config2)
                 (not (AtConf ?config1)) ) 
  )
  
)