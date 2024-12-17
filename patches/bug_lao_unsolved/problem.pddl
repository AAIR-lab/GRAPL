
(define (domain domain-t0)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types type0-t)
(:predicates (pred0-p ?v0 - type0-t ?v1 - type0-t) (pred1-p ?v0 - type0-t ?v1 - type0-t) (pred2-p ?v0 - type0-t))


	(:action action0-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred2-p ?x1))
		:effect 
		    (probabilistic  
		        0.700000 (and (not (pred2-p ?x0)))
		        0.300000 (and (not (pred2-p ?x0)) (not (pred1-p ?x0 ?x1)) (pred0-p ?x1 ?x0)))
	)


	(:action action3-a
		:parameters (?x0 - type0-t ?x1 - type0-t)
		:precondition 
			(and (pred1-p ?x0 ?x1))
		:effect 
		(probabilistic 
		    0.700000 (and (pred0-p ?x0 ?x1) (pred2-p ?x0) (pred2-p ?x1)) 
		    0.300000 (and (pred0-p ?x0 ?x1) (pred2-p ?x0) (not (pred1-p ?x1 ?x0))))
	)
)
(define (problem None) (:domain domain-t0)
  (:objects
        obj0-o - type0-t
	obj1-o - type0-t
	obj2-o - type0-t
	obj3-o - type0-t
  )
  (:init 
	(pred0-p obj0-o obj2-o)
	(pred1-p obj0-o obj2-o)
	(pred2-p obj2-o)
  )
  (:goal (and
	(pred0-p obj2-o obj0-o)
	(not (pred1-p obj0-o obj2-o))
	(pred2-p obj2-o)
	(pred0-p obj0-o obj1-o)
	(pred0-p obj0-o obj2-o)))
)
