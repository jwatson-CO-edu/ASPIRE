(define (domain pick-and-place)
  (:requirements :strips )

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;; Symbols ;;;
    (GraspObj ?label ?obj)
    (SafeMotion ?obj1 ?obj2 ?traj) ; Is there a safe path from config A to config B?: Checked by world
    (SafeCarry ?label ?obj1 ?obj2 ?traj)
    (StackPlace ?objUp ?objDn1 ?objDn2)

    ;;; Domains ;;;
    (Graspable ?label); Name of a real object we can grasp
    (Waypoint ?obj) ; Model of any object we can go to in the world, real or not
    ; (Occupied ?obj) ; Model of a real object we can grasp
    
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
                    (GraspObj ?label ?obj)
                    (AtObj ?obj)
                    (HandEmpty)
                    ; (Occupied ?obj)
                    (FreePlacement ?label ?obj) ; This is silly but the solver REQUIRES it!
                  )
    :effect (and (Holding ?label) 
                 (not (HandEmpty))
            )
    )

  (:action move_holding
    :parameters (?label ?obj1 ?obj2 ?traj)
    :precondition (and 
                    ;; Knowns ;;
                    (Holding ?label)
                    (AtObj ?obj1)
                    ;; Requirements ;;
                    (GraspObj ?label ?obj1)
                    (SafeCarry ?label ?obj1 ?obj2 ?traj)
                  )
    :effect (and (GraspObj ?label ?obj2)
                 (not (GraspObj ?label ?obj1))
                ;  (FreePlacement ?label ?obj1)
                ;  (not (FreePlacement ?label ?obj2))
                 (AtObj ?obj2)
                 (not (AtObj ?obj1))
                ;  (Occupied ?obj2)
                ;  (not (Occupied ?obj1))
            )
  )

  (:action place
    :parameters (?label ?obj)
    :precondition (and 
                    ;; Knowns ;;
                    (AtObj ?obj)
                    (Holding ?label)
                    ; (not (Occupied ?obj))
                    ;; Requirements ;;
                    (FreePlacement ?label ?obj)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?label)) 
                 (not (FreePlacement ?label ?obj))
                ;  (Block ?obj)
            )
  )
  

  (:action stack
    :parameters (?labelUp ?labelDn1 ?labelDn2 ?objUp ?objDn1 ?objDn2)
    :precondition (and 
                    ;; Knowns ;;
                    (AtObj ?objUp)
                    (Holding ?labelUp)
                    ; (not (Occupied ?objUp))
                    ;; Requirements ;;
                    (FreePlacement ?labelUp ?objUp)
                    ; (Occupied ?objDn1)
                    ; (Occupied ?objDn2)
                    (GraspObj ?labelDn1 ?objDn1)
                    (GraspObj ?labelDn2 ?objDn2)
                    (StackPlace ?objUp ?objDn1 ?objDn2)
                  )
    :effect (and (HandEmpty) 
                 (not (Holding ?labelUp))
                 (Supported ?labelUp ?labelDn1)
                 (Supported ?labelUp ?labelDn2)
                 (not (FreePlacement ?labelUp ?objUp)) ; Planner does dumb things if this isn't here
            )
  )
)  