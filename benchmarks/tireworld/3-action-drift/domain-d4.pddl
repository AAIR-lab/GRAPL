
(define (domain domain-d4)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types location)
(:predicates (vehicle-at ?v0 - location) (spare-in ?v0 - location) (road ?v0 - location ?v1 - location) (not-flattire))


	(:action move-car
		:parameters (?from - location ?to - location)
		:precondition 
			(and (vehicle-at ?from)
			(road ?from ?to)
			(not-flattire))
		:effect (and  (probabilistic 0.800000 (and
			(not (not-flattire))
			(not (vehicle-at ?from))) 0.200000 (and
			(not (road ?from ?to))
			(not (road ?from ?to))
			(not (road ?to ?from))
			(not (spare-in ?from))
			(vehicle-at ?to))))
	)


	(:action changetire
		:parameters (?loc - location)
		:precondition 
			(and (spare-in ?loc)
			(vehicle-at ?loc)
			(not (not-flattire)))
		:effect (and  (not (spare-in ?loc)) (not (vehicle-at ?loc)) (not-flattire) (probabilistic 1.000000 (and
			)))
	)
)