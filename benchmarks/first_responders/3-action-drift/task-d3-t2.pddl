
(define (problem task-d3-t2) (:domain domain-d3)
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
	(have-victim-in-unit z2 m1)
	(hospital l2)
	(medical-unit-at m1 l1)
	(medical-unit-at m1 l2)
	(victim-at z1 l2)
	(victim-at z2 l2)
	(victim-dying z2)
	(victim-hurt z1)
	(water-at l1)
  )
  (:goal (and
	(not (victim-at z1 l2))
	(not (victim-at z2 l2))
	(victim-healthy z2)
	(adjacent l2 l1)
	(not (medical-unit-at m1 l2))
	(victim-at z2 l1)
	(hospital l2)
	(victim-healthy z1)
	(fire-unit-at f1 l2)
	(medical-unit-at m1 l1)
	(not (victim-dying z2))
	(not (fire-unit-at f1 l1))
	(adjacent l1 l2)))
)
