(define
	(problem monkey)
	(:domain monkey)
	(:objects
		fruitBANANA - fruit
		locNORTH locEAST locSOUTH locWEST locCENTER - loc
	)
	(:init (chair-at locNORTH) (monkey-at locSOUTH) (fruit-at fruitBANANA locCENTER) (monkey-hungry))
	(:goal (not (monkey-hungry)))
)
