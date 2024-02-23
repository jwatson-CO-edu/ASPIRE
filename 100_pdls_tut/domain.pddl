(define
	(domain monkey)
	(:requirements :strips :typing)
	(:types
		fruit
		loc
	)
	(:predicates
		(chair-at ?loc - loc)
		(fruit-at ?fruit - fruit ?loc - loc)
		(monkey-at ?loc - loc)
		(monkey-hungry )
	)
	(:action eat
		:parameters (?fruit - fruit ?loc - loc)
		:precondition (and (monkey-at ?loc) (chair-at ?loc) (fruit-at ?fruit ?loc))
		:effect (and (not (monkey-hungry )) (not (fruit-at ?fruit ?loc)))
	)
	(:action go-from-to
		:parameters (?locBgn - loc ?locEnd - loc)
		:precondition (monkey-at ?locBgn)
		:effect (and (monkey-at ?locEnd) (not (monkey-at ?locBgn)))
	)
	(:action move-chair
		:parameters (?locBgn - loc ?locEnd - loc)
		:precondition (and (chair-at ?locBgn) (monkey-at ?locBgn))
		:effect (and (monkey-at ?locEnd) (not (monkey-at ?locBgn)) (chair-at ?locEnd) (not (chair-at ?locBgn)))
	)
)