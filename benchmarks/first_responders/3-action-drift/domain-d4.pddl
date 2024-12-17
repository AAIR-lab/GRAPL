
(define (domain domain-d4)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types location victim fire_unit medical_unit)
(:predicates (fire ?v0 - location) (nfire ?v0 - location) (victim-at ?v0 - victim ?v1 - location) (victim-healthy ?v0 - victim) (victim-hurt ?v0 - victim) (victim-dying ?v0 - victim) (hospital ?v0 - location) (water-at ?v0 - location) (adjacent ?v0 - location ?v1 - location) (fire-unit-at ?v0 - fire_unit ?v1 - location) (medical-unit-at ?v0 - medical_unit ?v1 - location) (have-water ?v0 - fire_unit) (have-victim-in-unit ?v0 - victim ?v1 - medical_unit))


	(:action drive-fire-unit
		:parameters (?u - fire_unit ?from - location ?to - location)
		:precondition 
			(and (fire-unit-at ?u ?from)
			(adjacent ?to ?from)
			(not (fire ?to)))
		:effect (and  (not (fire-unit-at ?u ?from)) (fire-unit-at ?u ?to) (probabilistic 1.000000 (and
			)))
	)


	(:action drive-medical-unit
		:parameters (?u - medical_unit ?from - location ?to - location)
		:precondition 
			(and (medical-unit-at ?u ?from)
			(adjacent ?to ?from)
			(not (fire ?to)))
		:effect (and  (not (medical-unit-at ?u ?from)) (medical-unit-at ?u ?to) (probabilistic 1.000000 (and
			)))
	)


	(:action load-fire-unit
		:parameters (?u - fire_unit ?l - location)
		:precondition 
			(and (not (have-water ?u))
			(not (water-at ?l))
			(not (fire-unit-at ?u ?l))
			(fire ?l))
		:effect (and  (not (fire ?l)) (not (nfire ?l)) (have-water ?u) (water-at ?l) (probabilistic 1.000000 (and
			)))
	)


	(:action load-medical-unit
		:parameters (?u - medical_unit ?l - location ?v - victim)
		:precondition 
			(and (medical-unit-at ?u ?l)
			(victim-at ?v ?l))
		:effect (and  (not (victim-at ?v ?l)) (probabilistic 1.000000 (and
			)))
	)


	(:action unload-fire-unit
		:parameters (?u - fire_unit ?l - location ?l1 - location)
		:precondition 
			(and (fire-unit-at ?u ?l)
			(adjacent ?l1 ?l)
			(have-water ?u)
			(fire ?l1))
		:effect (and  (not (fire ?l1)) (not (have-water ?u)) (probabilistic 1.000000 (and
			)))
	)


	(:action unload-medical-unit
		:parameters (?u - medical_unit ?l - location ?v - victim)
		:precondition 
			(and (medical-unit-at ?u ?l)
			(have-victim-in-unit ?v ?u))
		:effect (and  (not (have-victim-in-unit ?v ?u)) (victim-at ?v ?l) (probabilistic 1.000000 (and
			)))
	)


	(:action treat-victim-on-scene-medical
		:parameters (?u - medical_unit ?l - location ?v - victim)
		:precondition 
			(and (medical-unit-at ?u ?l)
			(victim-at ?v ?l)
			(victim-hurt ?v))
		:effect (and  (not (victim-hurt ?v)) (probabilistic 0.900000 (and
			(victim-healthy ?v)) 0.100000 (and
			(victim-dying ?v))))
	)


	(:action treat-victim-on-scene-fire
		:parameters (?u - fire_unit ?l - location ?v - victim)
		:precondition 
			(and (fire-unit-at ?u ?l)
			(victim-at ?v ?l)
			(victim-hurt ?v))
		:effect (and  (not (victim-hurt ?v)) (probabilistic 0.700000 (and
			(victim-healthy ?v)) 0.300000 (and
			(victim-dying ?v))))
	)


	(:action treat-hurt-victim-at-hospital
		:parameters (?v - victim ?l - location)
		:precondition 
			(and (victim-at ?v ?l)
			(victim-hurt ?v)
			(hospital ?l))
		:effect (and  (not (victim-hurt ?v)) (victim-healthy ?v) (probabilistic 1.000000 (and
			)))
	)


	(:action treat-dying-victim-at-hospital
		:parameters (?v - victim ?l - location)
		:precondition 
			(and (victim-at ?v ?l)
			(victim-dying ?v)
			(hospital ?l))
		:effect (and  (not (victim-dying ?v)) (victim-healthy ?v) (probabilistic 1.000000 (and
			)))
	)
)