
(define (domain domain-t0)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types type0-t)
(:predicates (pred0-p ?v0 - type0-t) (pred1-p ?v0 - type0-t) (pred2-p ?v0 - type0-t ?v1 - type0-t))


	(:action action0-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred0-p ?x0)
			(not (pred1-p ?x1)))
		:effect (and  (pred0-p ?x1) (pred1-p ?x1) (probabilistic 1.000000 (and
			)))
	)


	(:action action1-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred0-p ?x0)
			(not (pred1-p ?x1)))
		:effect (and  (pred1-p ?x1) (pred2-p ?x1 ?x0) (probabilistic 1.000000 (and
			)))
	)


	(:action action2-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred0-p ?x0))
		:effect (and  (not (pred1-p ?x0)) (pred2-p ?x0 ?x1) (probabilistic 1.000000 (and
			)))
	)
)