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
    (Obj ?label ?pose)
    (Grasp ?label ?pose ?effPose) ; A grasp has to have a target
    (IKSoln ?effPose ?config) ; Sample from hand pose
    (Path ?label ?poseBgn ?poseEnd ?traj) ; Is there a safe path from A to B?: Checked by world
    (StackPlace ?labelUp ?poseUp ?poseDn1 ?poseDn2)

    ;;; Domains ;;;
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
    (FreePlacement ?label ?pose) ; Is there an open spot for placement?: Checked by world
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
    :parameters (?label ?pose ?effPose ?config)
    :precondition (and 
                    ;; Knowns ;;
                    (Obj ?label ?pose)
                    (AtPose ?effPose)
                    (HandEmpty)
                    ;; Requirements ;;
                    (Grasp ?label ?pose ?effPose)
                  )
    :effect (and (Holding ?label) 
                 (not (HandEmpty)))
  )

  (:action move_holding
    :parameters (?label ?poseBgn ?poseEnd ?effPose1 ?effPose2 ?config1 ?config2 ?traj)
    :precondition (and 
                    ;; Knowns ;;
                    (Holding ?label)
                    (AtPose ?effPose1)
                    ;; Requirements ;;
                    (Obj ?label ?poseBgn)
                    (Grasp ?label ?poseBgn ?effPose1)   
                    (Grasp ?label ?poseEnd ?effPose2)  
                    (IKSoln ?effPose1 ?config1)
                    (IKSoln ?effPose2 ?config2) 
                    (SafeMotion ?config1 ?config2)
                    (Path ?label ?poseBgn ?poseEnd ?traj)
                  )
    :effect (and (Obj ?label ?poseEnd)
                 (not (Obj ?label ?poseBgn))
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
                    ; (Obj ?label ?pose)
                    ;; Requirements ;;
                    (FreePlacement ?label ?pose)
                    (Grasp ?label ?pose ?effPose)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?label)) ;)
                 (not (FreePlacement ?label ?pose)))
  )  

  (:action stack
    :parameters (?labelUp ?labelDn1 ?labelDn2 ?poseDn1 ?poseDn2 ?poseUp ?effPose)
    :precondition (and 
                    ;; Knowns ;;
                    (AtPose ?effPose)
                    (Holding ?labelUp)
                    ;; Requirements ;;
                    (Obj ?labelDn1 ?poseDn1)
                    (Obj ?labelDn2 ?poseDn2)
                    (StackPlace ?labelUp ?poseUp ?poseDn1 ?poseDn2)
                    (Grasp ?labelUp ?poseUp ?effPose)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?labelUp))
                 (Supported ?labelUp ?labelDn1)
                 (Supported ?labelUp ?labelDn2)
                ;  (not (FreePlacement ?labelUp ?poseUp))
            )
  )  
)