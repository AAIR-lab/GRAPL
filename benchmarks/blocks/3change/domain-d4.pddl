
(define (domain domain-d4)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types blocks)
(:predicates (clear ?v0 - blocks) (on-table ?v0 - blocks) (arm-empty) (holding ?v0 - blocks) (on ?v0 - blocks ?v1 - blocks))


	(:action pickup
		:parameters (?ob - blocks)
		:precondition 
			(and (on-table ?ob)
			(not (clear ?ob)))
		:effect (and  (not (arm-empty)) (not (on-table ?ob)) (clear ?ob) (probabilistic 1.000000 (and
			)))
	)


	(:action putdown
		:parameters (?ob - blocks)
		:precondition 
			(and (holding ?ob))
		:effect (and  (not (holding ?ob)) (not (on-table ?ob)) (arm-empty) (clear ?ob) (probabilistic 1.000000 (and
			)))
	)


	(:action stack
		:parameters (?ob - blocks ?underob - blocks)
		:precondition 
			(and (clear ?underob)
			(holding ?ob)
			(not (on ?underob ?ob)))
		:effect (and  (not (holding ?ob)) (arm-empty) (clear ?ob) (probabilistic 0.800000 (and
			(not (clear ?underob))) 0.200000 (and
			(not (on-table ?ob)))))
	)


	(:action unstack
		:parameters (?ob - blocks ?underob - blocks)
		:precondition 
			(and (on ?ob ?underob)
			(clear ?ob)
			(arm-empty))
		:effect (and  (not (arm-empty)) (not (on ?ob ?underob)) (clear ?underob) (holding ?ob) (probabilistic 1.000000 (and
			)))
	)
)