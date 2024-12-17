(define (problem fr-3)
 (:domain first-response)
 (:objects  l1 l2 l3 l4  - location
	    f1 - fire_unit
	    z1 z2 z3 - victim
	    m1 m2 m3 - medical_unit
)
 (:init 
	;;strategic locations
     (hospital l4)
     (hospital l1)
     (water-at l4)
	;;disaster info
     (fire l4)
     (victim-at z1 l1)
     (victim-dying z1)
     (fire l4)
     (victim-at z2 l4)
     (victim-dying z2)
     (fire l3)
     (victim-at z3 l4)
     (victim-hurt z3)
	;;map info
   (adjacent l1 l2)
   (adjacent l2 l1)
   (adjacent l3 l1)
   (adjacent l1 l3)
   (adjacent l3 l2)
   (adjacent l2 l3)
   (adjacent l4 l1)
   (adjacent l1 l4)
	(fire-unit-at f1 l4)
	(medical-unit-at m1 l1)
	(medical-unit-at m2 l1)
	(medical-unit-at m3 l2)
	)
 (:goal (and  (nfire l4)  (nfire l3)  (victim-healthy z1) (victim-healthy z2) (victim-healthy z3)))
 )
