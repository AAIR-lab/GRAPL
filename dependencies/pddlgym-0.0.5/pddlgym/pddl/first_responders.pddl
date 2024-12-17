(define (domain first-response)

    (:requirements :typing :negative-preconditions :non-deterministic)
    
    (:types location victim fire_unit medical_unit)
 
    (:predicates 
        (fire ?l - location)
        (nfire ?l - location)
        (victim-at ?v - victim ?l - location)
        
        (victim-healthy ?v - victim)
        (victim-hurt ?v - victim)
        (victim-dying ?v - victim)
        
        (hospital ?l - location)
        (water-at ?l - location)
        (adjacent ?l1 - location ?l2 - location)
        (fire-unit-at ?u - fire_unit ?l - location)
        (medical-unit-at ?u - medical_unit ?l - location)
        (have-water ?u - fire_unit)
        (have-victim-in-unit ?v - victim ?u - medical_unit)
        
        (drive-fire-unit ?u - fire_unit ?from - location ?to - location)
        (drive-medical-unit ?u - medical_unit ?from - location ?to - location)
        (load-fire-unit ?u - fire_unit ?l - location)
        (load-medical-unit ?u - medical_unit ?l - location ?v - victim)
        (unload-fire-unit ?u - fire_unit ?l - location ?l1 - location)
        (unload-medical-unit ?u - medical_unit ?l - location ?v - victim)
        (treat-victim-on-scene-medical ?u - medical_unit ?l - location ?v - victim)
        (treat-victim-on-scene-fire ?u - fire_unit ?l - location ?v - victim)
        (treat-hurt-victim-at-hospital ?v - victim ?l - location)
        (treat-dying-victim-at-hospital ?v - victim ?l - location)
  )

 ; (:actions drive-fire-unit drive-medical-unit load-fire-unit load-medical-unit unload-fire-unit unload-medical-unit treat-victim-on-scene-medical treat-victim-on-scene-fire treat-hurt-victim-at-hospital treat-dying-victim-at-hospital)

 (:action drive-fire-unit
  :parameters (?u - fire_unit ?from - location ?to - location)
  :precondition (and (fire-unit-at ?u ?from)
		            (adjacent ?to ?from)
		            (not (fire ?to))
		            (drive-fire-unit ?u ?from ?to)
		        )
  :effect (and (fire-unit-at ?u ?to) (not (fire-unit-at ?u ?from)))
  )

 (:action drive-medical-unit
  :parameters (?u - medical_unit ?from - location ?to - location)
  :precondition (and (medical-unit-at ?u ?from)
		            (adjacent ?to ?from)
		            (not (fire ?to))
		            (drive-medical-unit ?u ?from ?to)
		        )
  :effect (and (medical-unit-at ?u ?to) (not (medical-unit-at ?u ?from)))
  )

 (:action load-fire-unit
  :parameters (?u - fire_unit ?l - location)
  :precondition (and (not (have-water ?u)) (fire-unit-at ?u ?l) (water-at ?l) (load-fire-unit ?u ?l))
  :effect (and (have-water ?u)))

 (:action load-medical-unit
  :parameters (?u - medical_unit ?l - location ?v - victim)
  :precondition (and (medical-unit-at ?u ?l) (victim-at ?v ?l) (load-medical-unit ?u ?l ?v))
  :effect (and  (have-victim-in-unit ?v ?u)
	            (not (victim-at ?v ?l))
	       )
  )

 (:action unload-fire-unit
  :parameters (?u - fire_unit ?l - location ?l1 - location)
  :precondition (and (fire-unit-at ?u ?l)
                     (adjacent ?l1 ?l)
                     (have-water ?u)
                     (fire ?l1)
                     (unload-fire-unit ?u ?l ?l1)
                )
  :effect (and (not (have-water ?u)) (nfire ?l1) (not (fire ?l1)))
  )

 (:action unload-medical-unit
  :parameters (?u - medical_unit ?l - location ?v - victim)
  :precondition (and    (medical-unit-at ?u ?l)
                        (have-victim-in-unit ?v ?u)
                        (unload-medical-unit ?u ?l ?v)
                )
  :effect (and (victim-at ?v ?l) (not (have-victim-in-unit ?v ?u))))


 (:action treat-victim-on-scene-medical
  :parameters (?u - medical_unit ?l - location ?v - victim)
  :precondition (and    (medical-unit-at ?u ?l)
		                (victim-at ?v ?l)
		                (victim-hurt ?v)
		                (treat-victim-on-scene-medical ?u ?l ?v)
		        )
  :effect   (and (not (victim-hurt ?v)) (probabilistic 0.9 (and (victim-healthy ?v))
                                                       0.1 (and (victim-dying ?v))
                 )
            )
  )

 (:action treat-victim-on-scene-fire
  :parameters (?u - fire_unit ?l - location ?v - victim)
  :precondition (and    (fire-unit-at ?u ?l)
		                (victim-at ?v ?l)
		                (victim-hurt ?v)
		                (treat-victim-on-scene-fire ?u ?l ?v)
		        )
  :effect   (and (not (victim-hurt ?v)) (probabilistic 0.7 (and (victim-healthy ?v))
                                                       0.3 (and (victim-dying ?v))
                 )
            )
  )

 (:action treat-hurt-victim-at-hospital
  :parameters (?v - victim ?l - location)
  :precondition (and    (victim-at ?v ?l)
			(victim-hurt ?v)
		                (hospital ?l)
		                (treat-hurt-victim-at-hospital ?v ?l)
		        )
  :effect (and  (victim-healthy ?v)
	            (not (victim-hurt ?v))))

 (:action treat-dying-victim-at-hospital
  :parameters (?v - victim ?l - location)
  :precondition (and    (victim-at ?v ?l)
			(victim-dying ?v)
		                (hospital ?l)
		                (treat-dying-victim-at-hospital ?v ?l)
		        )
  :effect (and  (victim-healthy ?v)
	            (not (victim-dying ?v))))
  
)
