(define
	(problem monkey)
	(:domain monkey)
	(:objects
		elevLO elevHI - elev
		fruitBANANA - fruit
		locNORTH locEAST locSOUTH locWEST locCENTER - loc
		supportFLOOR supportCHAIR - support
	)
	(:init (support-at supportCHAIR locNORTH) 
		   (support-at supportFLOOR locNORTH) 
		   (support-at supportFLOOR locEAST) 
		   (support-at supportFLOOR locSOUTH) 
		   (support-at supportFLOOR locWEST) 
		   (support-at supportFLOOR locCENTER) 
		   (monkey-at locSOUTH elevLO) 
		   (fruit-at fruitBANANA locCENTER elevHI) 
		   (support-height supportFLOOR elevLO) 
		   (support-height supportCHAIR elevHI) 
		   (monkey-hungry))
	(:goal (not (monkey-hungry)))
)
