(define (domain pick-and-place)
  (:requirements :strips )

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;; Symbols ;;;
    (Obj ?label ?obj)
    ; (Grasp ?label ?pose ?effPose) ; A grasp has to have a target
    ; (IKSoln ?effPose ?config) ; Sample from hand pose
    ; (Path ?label ?objBgn ?objEnd ?traj) ; Is there a safe path from A to B?: Checked by world
    (StackPlace ?labelUp ?objUp ?objDn1 ?objDn2)

    ;;; Domains ;;;
    (Graspable ?label); Name of a real object we can grasp
    (Waypoint ?obj) ; Model of any object we can go to in the world, real or not
    ; (Conf ?config) ; Used by "stream.pddl"
    ; (Pose ?pose) ; Used by "stream.pddl", Do NOT pollute this space!
    ; (EffPose ?pose) ; Used by "stream.pddl"
  
    ;;; States ;;;
    (AtObj ?obj)
    (Holding ?label) ; From Pick
    (HandEmpty) ; From Place
    (Supported ?labelUp ?labelDn) ; Is the up object on top of the down object?

    ;;; Checks ;;;
    (FreePlacement ?label ?obj) ; Is there an open spot for placement?: Checked by world
    (SafeMotion ?obj1 ?obj2 ?traj) ; Is there a safe path from config A to config B?: Checked by world
  )

  ;;;;;;;;;; ACTIONS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:action move_free
    :parameters (?obj1 ?obj2 ?traj)
    :precondition (and 
                    ;; Knowns ;;
                    (AtObj ?obj1)
                    (HandEmpty)
                    ;; Requirements ;;
                    (SafeMotion ?obj1 ?obj2 ?traj)
                  )
    :effect (and (AtObj ?obj2)
                 (not (AtObj ?obj1))
            ) 
  )

  (:action pick
    :parameters (?label ?obj)
    :precondition (and 
                    ;; Knowns ;;
                    (Obj ?label ?obj)
                    (AtObj ?obj)
                    (HandEmpty)
                  )
    :effect (and (Holding ?label) 
                 (not (HandEmpty)))
  )

  (:action move_holding
    :parameters (?label ?obj1 ?obj2 ?traj)
    :precondition (and 
                    ;; Knowns ;;
                    (Holding ?label)
                    (AtObj ?obj1)
                    ;; Requirements ;;
                    (Obj ?label ?obj1)
                    ; (FreePlacement ?label ?obj2)
                    (SafeMotion ?obj1 ?obj2 ?traj)
                  )
    :effect (and (Obj ?label ?obj2)
                 (not (Obj ?label ?obj1))
                 (AtObj ?obj2)
                 (not (AtObj ?obj1))
            )
  )

  (:action place
    :parameters (?label ?obj)
    :precondition (and 
                    ;; Knowns ;;
                    (AtObj ?obj)
                    (Holding ?label)
                    ;; Requirements ;;
                    (FreePlacement ?label ?obj)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?label)) 
                 (not (FreePlacement ?label ?obj)))
  )  

  (:action stack
    :parameters (?labelUp ?labelDn1 ?labelDn2 ?objUp ?objDn1 ?objDn2)
    :precondition (and 
                    ;; Knowns ;;
                    (AtObj ?objUp)
                    (Holding ?labelUp)
                    ;; Requirements ;;
                    (Obj ?labelDn1 ?objDn1)
                    (Obj ?labelDn2 ?objDn2)
                    (StackPlace ?labelUp ?objUp ?objDn1 ?objDn2)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?labelUp))
                 (Supported ?labelUp ?labelDn1)
                 (Supported ?labelUp ?labelDn2)
            )
  )  
)