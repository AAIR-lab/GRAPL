
(define (domain domain-t0)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types object)
(:predicates (achieved ?g - object) (executed-action))


	(:action achieve-goal-v1-a
		:parameters (?g - object)
		:precondition 
			(and (not (achieved ?g)))
		:effect (and (probabilistic 0.8 (and (achieved ?g) (not (executed-action)))
					    0.2 (and (executed-action))
			))
	)
	
	(:action achieve-goal-v2-a
		:parameters (?g - object)
		:precondition 
			(and (not (achieved ?g)))
		:effect (and (probabilistic 0.5 (and (achieved ?g) (not (executed-action)))
					    0.5 (and (executed-action)))
			))
	)
	
)
