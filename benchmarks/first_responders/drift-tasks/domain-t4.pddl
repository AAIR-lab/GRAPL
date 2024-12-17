
(define (domain domain-t4)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types location victim fire_unit medical_unit)
(:predicates (fire ?v0 - location) (nfire ?v0 - location) (victim-at ?v0 - victim ?v1 - location) (victim-healthy ?v0 - victim) (victim-hurt ?v0 - victim) (victim-dying ?v0 - victim) (hospital ?v0 - location) (water-at ?v0 - location) (adjacent ?v0 - location ?v1 - location) (fire-unit-at ?v0 - fire_unit ?v1 - location) (medical-unit-at ?v0 - medical_unit ?v1 - location) (have-water ?v0 - fire_unit) (have-victim-in-unit ?v0 - victim ?v1 - medical_unit))


	(:action drive-fire-unit
		:parameters (?u - fire_unit ?from - location ?to - location)
		:precondition 
			(and (not (fire-unit-at ?u ?from))
			(fire ?to))
		:effect (and  (fire-unit-at ?u ?to) (probabilistic 1.000000 (and
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
			(fire-unit-at ?u ?l)
			(fire ?l)
			(not (nfire ?l))
			(not (water-at ?l)))
		:effect (and  (not (fire ?l)) (not (nfire ?l)) (fire ?l) (have-water ?u) (water-at ?l) (probabilistic 1.000000 (and
			)))
	)


	(:action load-medical-unit
		:parameters (?u - medical_unit ?l - location ?v - victim)
		:precondition 
			(and (victim-at ?v ?l))
		:effect (and  (not (victim-at ?v ?l)) (probabilistic 1.000000 (and
			)))
	)


	(:action unload-fire-unit
		:parameters (?u - fire_unit ?l - location ?l1 - location)
		:precondition 
			(and (fire-unit-at ?u ?l)
			(fire ?l1)
			(water-at ?l))
		:effect (and  (not (fire ?l1)) (not (fire ?l)) (have-water ?u) (probabilistic 1.000000 (and
			)))
	)


	(:action unload-medical-unit
		:parameters (?u - medical_unit ?l - location ?v - victim)
		:precondition 
			(and (medical-unit-at ?u ?l)
			(not (have-victim-in-unit ?v ?u))
			(fire ?l))
		:effect (and  (not (medical-unit-at ?u ?l)) (not (victim-at ?v ?l)) (fire ?l) (have-victim-in-unit ?v ?u) (water-at ?l) (probabilistic 1.000000 (and
			)))
	)


	(:action treat-victim-on-scene-medical
		:parameters (?u - medical_unit ?l - location ?v - victim)
		:precondition 
			(and (victim-at ?v ?l)
			(not (medical-unit-at ?u ?l))
			(not (fire ?l)))
		:effect (and  (probabilistic 0.900000 (and
			(victim-healthy ?v)
			(victim-hurt ?v)) 0.100000 (and
			(not (victim-dying ?v))
			(medical-unit-at ?u ?l)
			(medical-unit-at ?u ?l))))
	)


	(:action treat-victim-on-scene-fire
		:parameters (?u - fire_unit ?l - location ?v - victim)
		:precondition 
			(and (fire-unit-at ?u ?l)
			(not (fire ?l))
			(not (victim-at ?v ?l)))
		:effect (and  (not (victim-hurt ?v)) (fire ?l) (probabilistic 0.700000 (and
			(victim-healthy ?v)) 0.300000 (and
			(not (victim-dying ?v))
			(not (water-at ?l)))))
	)


	(:action treat-hurt-victim-at-hospital
		:parameters (?v - victim ?l - location)
		:precondition 
			(and (victim-at ?v ?l)
			(victim-hurt ?v)
			(hospital ?l)
			(not (fire ?l))
			(not (water-at ?l))
			(victim-dying ?v))
		:effect (and  (not (victim-hurt ?v)) (victim-healthy ?v) (water-at ?l) (probabilistic 1.000000 (and
			)))
	)


	(:action treat-dying-victim-at-hospital
		:parameters (?v - victim ?l - location)
		:precondition 
			(and (victim-dying ?v)
			(hospital ?l)
			(not (fire ?l))
			(not (water-at ?l))
			(not (victim-at ?v ?l)))
		:effect (and  (not (victim-healthy ?v)) (victim-dying ?v) (probabilistic 1.000000 (and
			)))
	)
)
; Drifted actions
;unload-medical-unit
;load-fire-unit
;treat-hurt-victim-at-hospital
;treat-victim-on-scene-medical
