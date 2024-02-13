(define
	(domain monkey)
	(:requirements :strips :typing)
	(:types
		elev
		fruit
		loc
		support
	)
	(:predicates
		(fruit-at ?fruit - fruit ?loc - loc ?elev - elev)
		(monkey-at ?loc - loc ?elev - elev)
		(monkey-hungry )
		(support-at ?support - support ?loc - loc)
		(support-height ?support - support ?elev - elev)
	)
	(:action eat
		:parameters (?fruit - fruit ?loc - loc ?elev - elev)
		:precondition (and (monkey-at ?loc ?elev) (fruit-at ?fruit ?loc ?elev))
		:effect (not (monkey-hungry ))
	)
	(:action go-from-to
		:parameters (?locBgn - loc ?elevBgn - elev ?support - support ?locEnd - loc ?elevEnd - elev)
		:precondition (and (monkey-at ?locBgn ?elevBgn) (support-at ?support ?locEnd) (support-height ?support ?elevEnd))
		:effect (and (monkey-at ?locEnd ?elevEnd) (not (monkey-at ?locBgn ?elevBgn)))
	)
	(:action move-support
		:parameters (?support - support ?locBgn - loc ?locEnd - loc)
		:precondition (support-at ?support ?locBgn)
		:effect (and (support-at ?support ?locEnd) (not (support-at ?support ?locBgn)))
	)
)