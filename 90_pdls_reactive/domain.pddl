(define (domain pick-and-place)
  (:requirements :strips )

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;;;; DEV PLAN ;;;;;
    ; [Y] Obj as a compact symbol type, 2024-02-09: I think it's done
    ; [Y] Grasp as particular to Obj, 2024-02-09: Seems correct
    ; [ ] Obj erased/created by moves
    ; [ ] Obj predicates validated by onboard poses, not "loose" poses

    ;;; Symbols ;;;
    (Grasp ?tgt ?effPose) ; A grasp has to have a target
    (IKSoln ?effPose ?config) ; Sample from hand pose
    (Path ?label ?bgnPose ?endPose ?traj) ; Is there a safe path from A to B?: Checked by world
    (StackPlace ?poseUp ?tgtDn1 ?tgtDn2)

    ;;; Domains ;;;
    (Obj ?obj) ; An object where it presently is in the world
    (Tgt ?tgt) ; An object where we wish it to be in the world
    (Graspable ?label) ; Used by "stream.pddl"
    (Conf ?config) ; Used by "stream.pddl"
    (Pose ?pose) ; Used by "stream.pddl", Do NOT pollute this space!
    (EffPose ?pose) ; Used by "stream.pddl"
  
    ;;; States ;;;
    (Holding ?label) ; From Pick
    (HandEmpty) ; From Place
    (AtConf ?config) ; From moves
    (AtPose ?effPose) ; From Move Holding
    (Supported ?labelUp ?labelDn) ; Is the up object on top of the down object?

    ;;; Checks ;;;
    ; (FreePlacement ?label ?pose) ; Is there an open spot for placement?: Checked by world
    (FreePlacement ?tgtDn1) ; Is there an open spot for placement?: Checked by world
    ; (SafeTransit ?label ?bgnPose ?endPose ) 
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
                 (not (HandEmpty))
                ;  (FreePlacement ?label ?pose))
                 (FreePlacement ?pose))
  )

  (:action move_holding
    :parameters (?label ?bgnPose ?endPose ?effPose1 ?effPose2 ?config1 ?config2 ?traj)
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
                    (Path ?label ?bgnPose ?endPose ?traj)
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
                    ; (FreePlacement ?label ?pose)
                    (FreePlacement ?pose)
                    (Grasp ?pose ?effPose)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?label))
                 (not (FreePlacement ?pose)))
  )  

  (:action stack
    :parameters (?labelUp ?labelDn1 ?labelDn2 ?poseUp ?poseDn1 ?poseDn2 ?effPose)
    ; :parameters (?poseUp ?labelUp ?labelDn1 ?labelDn2 ?effPose)
    :precondition (and 
                    ;; Knowns ;;
                    (AtPose ?effPose)
                    (Holding ?labelUp)
                    ;; Requirements ;;
                    (WObject ?labelDn1 ?poseDn1)
                    (WObject ?labelDn2 ?poseDn2)
                    (StackPlace ?labelDn1 ?labelDn2 ?poseUp ?poseDn1 ?poseDn2)
                    (FreePlacement ?poseUp)
                    (Grasp ?poseUp ?effPose)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?labelUp))
                 (Supported ?labelUp ?labelDn1)
                 (Supported ?labelUp ?labelDn2)
                 (not (FreePlacement ?poseUp))
            )
  )  
)