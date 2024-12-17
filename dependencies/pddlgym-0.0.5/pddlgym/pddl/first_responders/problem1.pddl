(define (problem problem1)
 (:domain first-response)
 (:objects  l1 l2   - location
	    f1 - fire_unit
	    z1 z2 - victim
	    m1 - medical_unit
)
 (:init
    ;;strategic locations
    (hospital l2)
    (water-at l1)

    ;;disaster info
    (fire l1)
    (fire l2)

    (victim-at z1 l2)
    (victim-hurt z1)
    (victim-at z2 l2)
    (victim-dying z2)

    ;;map info
    (adjacent l1 l2)
    (adjacent l2 l1)

	(fire-unit-at f1 l1)
	(medical-unit-at m1 l2)
	)
 (:goal (and  (nfire l1) (nfire l2) (victim-healthy z1) (victim-healthy z2)))
 )
