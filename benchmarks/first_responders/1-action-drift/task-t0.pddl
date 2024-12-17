
(define (problem task-t0) (:domain domain-t0)
  (:objects
        f1 - fire_unit
	l1 - location
	l2 - location
	m1 - medical_unit
	z1 - victim
	z2 - victim
  )
  (:init 
	(adjacent l1 l2)
	(adjacent l2 l1)
	(fire l1)
	(fire l2)
	(fire-unit-at f1 l1)
	(hospital l2)
	(medical-unit-at m1 l2)
	(victim-at z1 l2)
	(victim-at z2 l2)
	(victim-dying z2)
	(victim-hurt z1)
	(water-at l1)
  )
  (:goal (and
	(nfire l1)
	(nfire l2)
	(victim-healthy z1)
	(victim-healthy z2)))
)
