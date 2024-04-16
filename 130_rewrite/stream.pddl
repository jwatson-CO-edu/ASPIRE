(define (stream magpie-tamp)

;;;;;;;;;; SYMBOL STREAMS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ; Stacking Pose Stream ;;
  (:stream sample-above
    :inputs (?label)
    :domain (Graspable ?label)
    ; :fluents (Free) 
    :outputs (?pose)
    :certified (and (PoseAbove ?pose ?label) (Waypoint ?pose) (Free ?pose) )
  )

;;;;;;;;;; TESTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ; ; Free Placement Test ;;
  ; (:stream test-free-placment
  ;   :inputs (?pose)
  ;   :domain (Waypoint ?pose)
  ;   :certified (Free ?pose)
  ; )
)