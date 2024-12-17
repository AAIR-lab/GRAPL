
(define (problem task-t4) (:domain domain-t4)
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
	(fire-unit-at f1 l2)
	(have-victim-in-unit z1 m1)
	(have-victim-in-unit z2 m1)
	(hospital l1)
	(hospital l2)
	(medical-unit-at m1 l1)
	(medical-unit-at m1 l2)
	(victim-at z1 l1)
	(victim-at z1 l2)
	(victim-at z2 l1)
	(victim-at z2 l2)
	(victim-dying z1)
	(victim-dying z2)
	(victim-hurt z1)
	(victim-hurt z2)
	(water-at l1)
	(water-at l2)
  )
  (:goal (and
	(water-at l1)
	(not (medical-unit-at m1 l2))
	(hospital l2)
	(victim-healthy z1)
	(not (victim-dying z2))
	(adjacent l1 l2)
	(victim-at z1 l1)
	(water-at l2)
	(not (have-victim-in-unit z2 m1))
	(not (fire-unit-at f1 l2))
	(not (victim-at z1 l2))
	(fire-unit-at f1 l1)
	(victim-at z2 l2)
	(medical-unit-at m1 l1)
	(not (victim-dying z1))
	(have-water f1)
	(hospital l1)
	(victim-healthy z2)
	(adjacent l2 l1)
	(not (victim-at z2 l1))
	(not (have-victim-in-unit z1 m1))))
)