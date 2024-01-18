(define
	(problem blocks)
	(:domain blocks)
	(:objects
		redBlock yellowBlock blueBlock loc-a loc-b loc-c - object
	)
	(:init (Block redBlock) (Block blueBlock) (Block yellowBlock) (on yellowBlock loc-a) (on redBlock loc-c) (on blueBlock loc-b) (fixed loc-a) (fixed loc-b) (fixed loc-c) (clear yellowBlock) (clear redBlock) (clear blueBlock))
	(:goal (on blueBlock yellowBlock))
)
