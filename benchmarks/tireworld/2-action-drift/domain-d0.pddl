
(define (domain domain-d0)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types location)
(:predicates (vehicle-at ?v0 - location) (spare-in ?v0 - location) (road ?v0 - location ?v1 - location) (not-flattire))


	(:action move-car
		:parameters (?from - location ?to - location)
		:precondition 
			(and (vehicle-at ?from)
			(road ?from ?to)
			(not-flattire))
		:effect (and  (not (vehicle-at ?from)) (vehicle-at ?to) (probabilistic 0.800000 (and
			(not (not-flattire))) 0.200000 (and
			)))
	)


	(:action changetire
		:parameters (?loc - location)
		:precondition 
			(and (spare-in ?loc)
			(vehicle-at ?loc)
			(not (not-flattire)))
		:effect (and  (not (spare-in ?loc)) (not-flattire) (probabilistic 1.000000 (and
			)))
	)
)