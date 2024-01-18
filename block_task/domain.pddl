(define
	(domain blocks)
	(:requirements :strips :typing)
	(:types
		object
	)
	(:predicates
		(Block ?objectA - object)
		(clear ?objectA - object)
		(fixed ?objectA - object)
		(on ?objectA - object ?objectB - object)
	)
	(:action move
		:parameters (?block - object ?underObject - object ?newUnderObject - object)
		:precondition (and (not (fixed ?block)) (Block ?block) (on ?block ?underObject) (clear ?block) (clear ?newUnderObject))
		:effect (and (not (on ?block ?underObject)) (on ?block ?newUnderObject) (clear ?block) (clear ?underObject) (not (clear ?newUnderObject)))
	)
)