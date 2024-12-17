
(define (domain domain-t4)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types location)
(:predicates (vehicle-at ?v0 - location) (spare-in ?v0 - location) (road ?v0 - location ?v1 - location) (not-flattire))


	(:action move-car
		:parameters (?from - location ?to - location)
		:precondition 
			(and (vehicle-at ?from)
			(road ?from ?to)
			(spare-in ?to))
		:effect (and  (probabilistic 0.800000 (and
			(not (not-flattire))
			(vehicle-at ?from)
			(vehicle-at ?to)) 0.200000 (and
			(not (vehicle-at ?from)))))
	)


	(:action changetire
		:parameters (?loc - location)
		:precondition 
			(and (vehicle-at ?loc))
		:effect (and  (not (not-flattire)) (not (spare-in ?loc)) (probabilistic 1.000000 (and
			)))
	)
)