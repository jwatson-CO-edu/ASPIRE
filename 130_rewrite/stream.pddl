(define (stream magpie-tamp)

;;;;;;;;;; SYMBOL STREAMS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ; Stacking Pose Stream ;;
  (:stream sample-above
    :inputs (?label)
    :domain (Graspable ?label)
    :outputs (?pose)
    :certified (and (PoseAbove ?pose ?label) (Waypoint ?pose) ) ;(Free ?pose) ) ; 2024-04-19: This causes shoving?
    ; :certified (and (PoseAbove ?pose ?label) (Waypoint ?pose) (Free ?pose) ) ; 2024-04-19: This causes shoving?
  )

;;;;;;;;;; TESTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ; Free Placement Test ;;
  (:stream test-free-placment
    :inputs (?pose)
    :domain (Waypoint ?pose)
    :certified (Free ?pose)
  )
)