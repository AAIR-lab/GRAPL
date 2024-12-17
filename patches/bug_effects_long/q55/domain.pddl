
(define (domain domain-t0)
(:requirements :typing :strips :non-deterministic :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types type0-t)
(:predicates (pred0-p_1 ?v0 - type0-t ?v1 - type0-t) (pred0-p_2 ?v0 - type0-t ?v1 - type0-t) (pred1-p_1 ?v0 - type0-t) (pred1-p_2 ?v0 - type0-t) (pred2-p_1 ?v0 - type0-t ?v1 - type0-t) (pred2-p_2 ?v0 - type0-t ?v1 - type0-t) (action0-a_1 ?v0 - type0-t ?v1 - type0-t) (action0-a_2 ?v0 - type0-t ?v1 - type0-t) (action1-a_1 ?v0 - type0-t ?v1 - type0-t) (action1-a_2 ?v0 - type0-t ?v1 - type0-t) (action2-a_1 ?v0 - type0-t ?v1 - type0-t) (action2-a_2 ?v0 - type0-t ?v1 - type0-t) (action3-a_1 ?v0 - type0-t ?v1 - type0-t) (action3-a_2 ?v0 - type0-t ?v1 - type0-t) (p_psi))


	(:action action0-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition (and (not (= ?x0 ?x1)) 
			(and (not (pred1-p_1 ?x1))))
		:effect (and
			(when (and
			(not (pred1-p_1 ?x1))
			(not (pred1-p_2 ?x1))) (and
			(oneof (and
			(pred0-p_1 ?x0 ?x1)))
			(pred0-p_1 ?x0 ?x1)
			(oneof (and
			(pred0-p_2 ?x0 ?x1)))
			(not (pred0-p_2 ?x0 ?x1))))
			
				(when (pred1-p_2 ?x1) (and
			(p_psi)))
			
				(when (pred1-p_1 ?x1) (and
			(p_psi))))
	)


	(:action action1-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition (and (not (= ?x0 ?x1)) 
			(and (pred0-p_1 ?x1 ?x0)
			(pred2-p_1 ?x0 ?x1)))
		:effect (and
			(when (and
			(pred0-p_1 ?x1 ?x0)
			(pred2-p_1 ?x0 ?x1)
			(pred0-p_2 ?x1 ?x0)
			(pred2-p_2 ?x0 ?x1)) (and
			(oneof)
			(oneof))))
	)


	(:action action2-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition (and (not (= ?x0 ?x1)) 
			(and (pred0-p_1 ?x1 ?x0)
			(not (pred1-p_1 ?x1))))
		:effect (and
			(when (and
			(pred0-p_1 ?x1 ?x0)
			(not (pred1-p_1 ?x1))
			(pred0-p_2 ?x1 ?x0)
			(not (pred1-p_2 ?x1))) (and
			(oneof (and
			(pred1-p_1 ?x0)) (and
			(not (pred2-p_1 ?x0 ?x1))
			(not (pred0-p_1 ?x1 ?x0))))
			(oneof (and
			(pred1-p_2 ?x0)) (and
			(not (pred2-p_2 ?x0 ?x1))
			(not (pred0-p_2 ?x1 ?x0)))))))
	)


	(:action action3-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition (and (not (= ?x0 ?x1)) 
			(and (pred2-p_1 ?x1 ?x0)
			(pred1-p_1 ?x1)))
		:effect (and
			(when (and
			(pred2-p_1 ?x1 ?x0)
			(pred1-p_1 ?x1)
			(pred2-p_2 ?x1 ?x0)
			(pred1-p_2 ?x1)) (and
			(oneof (and
			(not (pred1-p_1 ?x1))
			(pred2-p_1 ?x0 ?x1)) (and
			(pred2-p_1 ?x0 ?x1)))
			(oneof (and
			(not (pred1-p_2 ?x1))
			(pred2-p_2 ?x0 ?x1)) (and
			(pred2-p_2 ?x0 ?x1))))))
	)


	(:action action0-a2
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition (and (not (= ?x0 ?x1)) 
			(and (not (pred1-p_2 ?x1))))
		:effect (and
			(when (and
			(not (pred1-p_1 ?x1))
			(not (pred1-p_2 ?x1))) (and
			(oneof (and
			(pred0-p_1 ?x0 ?x1)))
			(pred0-p_1 ?x0 ?x1)
			(oneof (and
			(pred0-p_2 ?x0 ?x1)))
			(not (pred0-p_2 ?x0 ?x1))))
			
				(when (pred1-p_2 ?x1) (and
			(p_psi)))
			
				(when (pred1-p_1 ?x1) (and
			(p_psi))))
	)


	(:action action0-a300
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition (and (not (= ?x0 ?x1)) 
			(and (pred0-p_1 ?x0 ?x1)
			(not (pred0-p_2 ?x0 ?x1))))
		:effect (and
			(p_psi))
	)


	(:action action1-a2
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition (and (not (= ?x0 ?x1)) 
			(and (pred0-p_2 ?x1 ?x0)
			(pred2-p_2 ?x0 ?x1)))
		:effect (and
			(when (and
			(pred0-p_1 ?x1 ?x0)
			(pred2-p_1 ?x0 ?x1)
			(pred0-p_2 ?x1 ?x0)
			(pred2-p_2 ?x0 ?x1)) (and
			(oneof)
			(oneof))))
	)


	(:action action2-a2
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition (and (not (= ?x0 ?x1)) 
			(and (pred0-p_2 ?x1 ?x0)
			(not (pred1-p_2 ?x1))))
		:effect (and
			(when (and
			(pred0-p_1 ?x1 ?x0)
			(not (pred1-p_1 ?x1))
			(pred0-p_2 ?x1 ?x0)
			(not (pred1-p_2 ?x1))) (and
			(oneof (and
			(pred1-p_1 ?x0)) (and
			(not (pred2-p_1 ?x0 ?x1))
			(not (pred0-p_1 ?x1 ?x0))))
			(oneof (and
			(pred1-p_2 ?x0)) (and
			(not (pred2-p_2 ?x0 ?x1))
			(not (pred0-p_2 ?x1 ?x0)))))))
	)


	(:action action3-a2
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition (and (not (= ?x0 ?x1)) 
			(and (pred2-p_2 ?x1 ?x0)
			(pred1-p_2 ?x1)))
		:effect (and
			(when (and
			(pred2-p_1 ?x1 ?x0)
			(pred1-p_1 ?x1)
			(pred2-p_2 ?x1 ?x0)
			(pred1-p_2 ?x1)) (and
			(oneof (and
			(not (pred1-p_1 ?x1))
			(pred2-p_1 ?x0 ?x1)) (and
			(pred2-p_1 ?x0 ?x1)))
			(oneof (and
			(not (pred1-p_2 ?x1))
			(pred2-p_2 ?x0 ?x1)) (and
			(pred2-p_2 ?x0 ?x1))))))
	)
)