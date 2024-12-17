
(define (domain domain-d3)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types blocks)
(:predicates (clear ?v0 - blocks) (on-table ?v0 - blocks) (arm-empty) (holding ?v0 - blocks) (on ?v0 - blocks ?v1 - blocks))


	(:action pickup
		:parameters (?ob - blocks)
		:precondition 
			(and (on-table ?ob)
			(arm-empty)
			(not (holding ?ob)))
		:effect (and  (not (arm-empty)) (not (clear ?ob)) (not (on-table ?ob)) (holding ?ob) (probabilistic 1.000000 (and
			)))
	)


	(:action putdown
		:parameters (?ob - blocks)
		:precondition 
			(and (holding ?ob))
		:effect (and  (not (holding ?ob)) (arm-empty) (clear ?ob) (on-table ?ob) (probabilistic 1.000000 (and
			)))
	)


	(:action stack
		:parameters (?ob - blocks ?underob - blocks)
		:precondition 
			(and (clear ?underob)
			(holding ?ob))
		:effect (and  (not (holding ?ob)) (arm-empty) (clear ?ob) (probabilistic 0.800000 (and
			(not (clear ?underob))
			(on ?ob ?underob)) 0.200000 (and
			(on-table ?ob))))
	)


	(:action unstack
		:parameters (?ob - blocks ?underob - blocks)
		:precondition 
			(and (clear ?underob)
			(not (on ?ob ?underob))
			(not (clear ?ob)))
		:effect (and  (not (arm-empty)) (not (clear ?underob)) (not (holding ?ob)) (not (holding ?underob)) (not (on ?underob ?ob)) (clear ?ob) (on ?ob ?underob) (on-table ?ob) (on-table ?underob) (probabilistic 1.000000 (and
			)))
	)
)