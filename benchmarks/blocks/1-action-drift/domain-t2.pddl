
(define (domain domain-t2)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types blocks)
(:predicates (clear ?v0 - blocks) (on-table ?v0 - blocks) (arm-empty) (holding ?v0 - blocks) (on ?v0 - blocks ?v1 - blocks))


	(:action pickup
		:parameters (?ob - blocks)
		:precondition 
			(and (clear ?ob)
			(on-table ?ob)
			(arm-empty))
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
			(and (clear ?ob)
			(not (on ?ob ?underob)))
		:effect (and  (not (arm-empty)) (not (clear ?ob)) (not (on ?underob ?ob)) (holding ?ob) (on ?ob ?underob) (probabilistic 1.000000 (and
			)))
	)
)
; Drifted actions
;unstack