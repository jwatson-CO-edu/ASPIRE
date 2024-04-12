(define (stream magpie-tamp)

;;;;;;;;;; SYMBOL STREAMS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ; Stacking Pose Stream ;;
  (:stream sample-above
    :inputs (?label)
    :domain (Graspable ?label)
    :outputs (?pose)
    :certified (and (Waypoint ?pose) (PoseAbove ?pose ?label) )
  )

;;;;;;;;;; TESTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ; ; Free Placement Test ;;
  ; (:stream test-free-placment
  ;   :inputs (?pose)
  ;   :domain (Waypoint ?pose)
  ;   :certified (Free ?pose)
  ; )
)