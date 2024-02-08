(define (domain pick-and-place)
  (:requirements :strips )

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;; Symbols ;;;
    (WObject ?label ?pose); Sample from world -or- Produced by actions
    (IKSoln ?pose ?config) ; Sample from hand pose
    (Grasp ?pose ?effPose) ; Sample from object pose

    ;;; Domains ;;;
    (Graspable ?label) ; Used by "stream.pddl"
    (Conf ?config) ; Used by "stream.pddl"
    (Pose ?pose) ; Used by "stream.pddl"
    (EffPose ?pose) ; Used by "stream.pddl"
  
    ;;; States ;;;
    (Holding ?label) ; From Pick
    (HandEmpty) ; From Place
    (AtConf ?config) ; From moves
    (AtPose ?effPose) ; From Move Holding

    ;;; Checks ;;;
    (FreePlacement ?label ?pose) ; Is there an open spot for placement?: Checked by world
    (SafeTransit ?label ?bgnPose ?endPose ) ; Is there a safe path from A to B?: Checked by world
    (SafeMotion ?config1 ?config2) ; Is there a safe path from config A to config B?: Checked by world
  )

  ;;;;;;;;;; ACTIONS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:action move_q
    :parameters (?config1 ?config2)
    :precondition (and 
                    ;; Knowns ;;
                    (AtConf ?config1) 
                    ;; Requirements ;;
                    (SafeMotion ?config1 ?config2)
                  )
    :effect (and (AtConf ?config2)
                 (not (AtConf ?config1)) ) 
  )

  (:action move_free
    :parameters (?effPose1 ?effPose2 ?config1 ?config2)
    :precondition (and 
                    ;; Knowns ;;
                    (AtPose ?effPose1)
                    (HandEmpty)
                    ;; Requirements ;;
                    (IKSoln ?effPose2 ?config2) 
                    (IKSoln ?effPose1 ?config1)
                    (SafeMotion ?config1 ?config2)
                  )
    :effect (and (AtPose ?effPose2)
                 (not (AtPose ?effPose1))
                 (AtConf ?config2)
                 (not (AtConf ?config1))
            ) 
  )

  (:action pick
    :parameters (?label ?pose ?effPose)
    :precondition (and 
                    ;; Knowns ;;
                    (WObject ?label ?pose)
                    (AtPose ?effPose)
                    (HandEmpty)
                    ;; Requirements ;;
                    (Grasp ?pose ?effPose)   
                  )
    :effect (and (Holding ?label) 
                 (not (HandEmpty)))
  )

  (:action move_holding
    :parameters (?label ?bgnPose ?endPose ?effPose1 ?effPose2 ?config1 ?config2)
    :precondition (and 
                    ;; Knowns ;;
                    (Holding ?label)
                    (AtPose ?effPose1)
                    ;; Requirements ;;
                    (Grasp ?bgnPose ?effPose1)
                    (IKSoln ?effPose1 ?config1)
                    (Grasp ?endPose ?effPose2)
                    (IKSoln ?effPose2 ?config2) 
                    (SafeMotion ?config1 ?config2)
                    (SafeTransit ?label ?bgnPose ?endPose)
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
    :precondition (and 
                    ;; Knowns ;;
                    (AtPose ?effPose)
                    (Holding ?label)
                    ;; Requirements ;;
                    (FreePlacement ?label ?pose)
                    (Grasp ?pose ?effPose)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?label)))
  )  
)