
(define (domain domain-t3)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types type0-t)
(:predicates (pred0-p ?v0 - type0-t ?v1 - type0-t) (pred1-p ?v0 - type0-t ?v1 - type0-t) (pred2-p ?v0 - type0-t ?v1 - type0-t) (pred3-p ?v0 - type0-t ?v1 - type0-t) (pred4-p ?v0 - type0-t ?v1 - type0-t))


	(:action action0-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred4-p ?x1 ?x0)
			(pred3-p ?x0 ?x1))
		:effect (and  (not (pred3-p ?x0 ?x1)) (not (pred4-p ?x1 ?x0)) (pred0-p ?x1 ?x0) (probabilistic 1.000000 (and
			)))
	)


	(:action action1-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred4-p ?x0 ?x1)
			(not (pred0-p ?x0 ?x1)))
		:effect (and  (not (pred0-p ?x1 ?x0)) (not (pred1-p ?x1 ?x0)) (not (pred4-p ?x0 ?x1)) (pred4-p ?x1 ?x0) (probabilistic 1.000000 (and
			)))
	)


	(:action action2-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred0-p ?x1 ?x0)
			(not (pred2-p ?x1 ?x0))
			(not (pred3-p ?x1 ?x0)))
		:effect (and  (not (pred1-p ?x0 ?x1)) (not (pred2-p ?x0 ?x1)) (pred2-p ?x1 ?x0) (pred4-p ?x0 ?x1) (probabilistic 1.000000 (and
			)))
	)


	(:action action3-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred4-p ?x0 ?x1)
			(pred3-p ?x0 ?x1))
		:effect (and  (not (pred0-p ?x0 ?x1)) (not (pred0-p ?x1 ?x0)) (not (pred2-p ?x1 ?x0)) (not (pred3-p ?x0 ?x1)) (probabilistic 1.000000 (and
			)))
	)


	(:action action4-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred2-p ?x1 ?x0)
			(pred3-p ?x1 ?x0))
		:effect (and  (not (pred1-p ?x0 ?x1)) (not (pred3-p ?x0 ?x1)) (not (pred3-p ?x1 ?x0)) (probabilistic 1.000000 (and
			)))
	)


	(:action action5-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred0-p ?x1 ?x0)
			(pred3-p ?x0 ?x1))
		:effect (and  (not (pred3-p ?x0 ?x1)) (pred1-p ?x1 ?x0) (pred2-p ?x0 ?x1) (probabilistic 1.000000 (and
			)))
	)


	(:action action6-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred3-p ?x1 ?x0)
			(pred4-p ?x0 ?x1))
		:effect (and  (not (pred4-p ?x0 ?x1)) (pred0-p ?x1 ?x0) (pred1-p ?x0 ?x1) (pred1-p ?x1 ?x0) (probabilistic 1.000000 (and
			)))
	)


	(:action action7-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred0-p ?x1 ?x0)
			(pred1-p ?x0 ?x1)
			(pred4-p ?x1 ?x0))
		:effect (and  (not (pred0-p ?x0 ?x1)) (not (pred4-p ?x1 ?x0)) (pred3-p ?x1 ?x0) (probabilistic 1.000000 (and
			)))
	)


	(:action action8-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (not (pred2-p ?x0 ?x1))
			(pred0-p ?x1 ?x0))
		:effect (and  (not (pred2-p ?x1 ?x0)) (not (pred3-p ?x0 ?x1)) (pred2-p ?x0 ?x1) (pred4-p ?x1 ?x0) (probabilistic 1.000000 (and
			)))
	)


	(:action action9-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred1-p ?x0 ?x1)
			(not (pred2-p ?x1 ?x0)))
		:effect (and  (not (pred1-p ?x0 ?x1)) (pred2-p ?x1 ?x0) (pred4-p ?x0 ?x1) (probabilistic 1.000000 (and
			)))
	)
)
; Drifted actions
;action6-a
