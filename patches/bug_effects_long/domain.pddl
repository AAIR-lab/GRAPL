
(define (domain domain-t0)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types type0-t)
(:predicates (pred0-p ?v0 - type0-t ?v1 - type0-t) (pred1-p ?v0 - type0-t) (pred2-p ?v0 - type0-t ?v1 - type0-t))


	(:action action0-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (not (pred1-p ?x1)))
		:effect (and  (probabilistic 1.000000 (and
			(not (pred2-p ?x0 ?x1))
			(not (pred2-p ?x1 ?x0))
			(pred0-p ?x0 ?x1))))
	)


	(:action action1-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred2-p ?x0 ?x1)
			(pred0-p ?x1 ?x0))
		:effect (and  (probabilistic 1.000000 (and
			(not (pred0-p ?x0 ?x1))
			(not (pred2-p ?x1 ?x0)))))
	)


	(:action action2-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (not (pred1-p ?x1))
			(pred0-p ?x1 ?x0))
		:effect (and  (probabilistic 0.700000 (and
			(not (pred0-p ?x1 ?x0))
			(not (pred1-p ?x0))
			(not (pred2-p ?x0 ?x1))) 0.300000 (and
			(pred1-p ?x0))))
	)


	(:action action3-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred1-p ?x1)
			(pred2-p ?x1 ?x0))
		:effect (and  (probabilistic 0.600000 (and
			(pred2-p ?x0 ?x1)) 0.400000 (and
			(not (pred0-p ?x1 ?x0))
			(not (pred1-p ?x1))
			(pred2-p ?x0 ?x1))))
	)
)