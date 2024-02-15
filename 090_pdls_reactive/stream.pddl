(define (stream magpie-tamp)

;;;;;;;;;; SYMBOL STREAMS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


  ; Object Pose Stream ;;
  (:stream sample-object
    :inputs (?label)
    :domain (Graspable ?label)
    :outputs (?obj)
    :certified (and (Obj ?label ?obj) (Waypoint ?obj))
  )

  ;; Safe Motion Planner ;;
  (:stream find-safe-motion
    :inputs (?obj1 ?obj2)
    :domain (and (Waypoint ?obj1) (Waypoint ?obj2))
    :outputs (?traj)
    :certified (SafeMotion ?obj1 ?obj2 ?traj)
  )

  ; ;; Stack Location Search ;;
  (:stream find-stack-place
    :inputs (?labelUp ?objDn1 ?objDn2)
    :domain (and (Graspable ?labelUp) (Waypoint ?objDn1) (Waypoint ?objDn2))
    :outputs (?objUp)
    :certified (and (StackPlace ?labelUp ?objUp ?objDn1 ?objDn2) (Waypoint ?objUp))
  )


;;;;;;;;;; TESTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ; Free Placement Search ;;
  (:stream test-free-placment
    :inputs (?label ?obj)
    :domain (and (Graspable ?label) (Waypoint ?obj))
    :certified (FreePlacement ?label ?obj)
  )
)