(define (problem FR_7_10)
 (:domain first-response)
 (:objects  l1 l2 l3 l4 l5 l6 l7  - location
	    f1 f2 - fire_unit
	    v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 - victim
	    m1 m2 m3 m4 - medical_unit
)
 (:init 
	;;strategic locations
     (hospital l3)
     (water-at l5)
	;;disaster info
     (fire l3)
     (victim-at v1 l2)
     (victim-status v1 hurt)
     (fire l3)
     (victim-at v2 l1)
     (victim-status v2 dying)
     (fire l3)
     (victim-at v3 l2)
     (victim-status v3 dying)
     (fire l3)
     (victim-at v4 l2)
     (victim-status v4 hurt)
     (fire l3)
     (victim-at v5 l2)
     (victim-status v5 hurt)
     (fire l3)
     (victim-at v6 l2)
     (victim-status v6 dying)
     (fire l3)
     (victim-at v7 l1)
     (victim-status v7 dying)
     (fire l3)
     (victim-at v8 l2)
     (victim-status v8 hurt)
     (fire l3)
     (victim-at v9 l1)
     (victim-status v9 hurt)
     (fire l3)
     (victim-at v10 l2)
     (victim-status v10 dying)
	;;map info
	(adjacent l1 l1)
	(adjacent l2 l2)
	(adjacent l3 l3)
	(adjacent l4 l4)
	(adjacent l5 l5)
	(adjacent l6 l6)
	(adjacent l7 l7)
   (adjacent l1 l1)
   (adjacent l1 l1)
   (adjacent l1 l2)
   (adjacent l2 l1)
   (adjacent l2 l1)
   (adjacent l1 l2)
   (adjacent l2 l2)
   (adjacent l2 l2)
   (adjacent l2 l3)
   (adjacent l3 l2)
   (adjacent l2 l4)
   (adjacent l4 l2)
   (adjacent l2 l5)
   (adjacent l5 l2)
   (adjacent l4 l1)
   (adjacent l1 l4)
   (adjacent l4 l2)
   (adjacent l2 l4)
   (adjacent l4 l3)
   (adjacent l3 l4)
   (adjacent l4 l4)
   (adjacent l4 l4)
   (adjacent l5 l1)
   (adjacent l1 l5)
   (adjacent l5 l2)
   (adjacent l2 l5)
   (adjacent l5 l3)
   (adjacent l3 l5)
   (adjacent l5 l4)
   (adjacent l4 l5)
   (adjacent l5 l5)
   (adjacent l5 l5)
   (adjacent l5 l6)
   (adjacent l6 l5)
   (adjacent l6 l1)
   (adjacent l1 l6)
   (adjacent l6 l2)
   (adjacent l2 l6)
   (adjacent l7 l1)
   (adjacent l1 l7)
   (adjacent l7 l2)
   (adjacent l2 l7)
   (adjacent l7 l3)
   (adjacent l3 l7)
   (adjacent l7 l4)
   (adjacent l4 l7)
   (adjacent l7 l5)
   (adjacent l5 l7)
	(fire-unit-at f1 l2)
	(fire-unit-at f2 l7)
	(medical-unit-at m1 l6)
	(medical-unit-at m2 l4)
	(medical-unit-at m3 l2)
	(medical-unit-at m4 l1)
	)
 (:goal (and  (nfire l3) (nfire l3) (nfire l3) (nfire l3) (nfire l3) (nfire l3) (nfire l3) (nfire l3) (nfire l3) (nfire l3)  (victim-status v1 healthy) (victim-status v2 healthy) (victim-status v3 healthy) (victim-status v4 healthy) (victim-status v5 healthy) (victim-status v6 healthy) (victim-status v7 healthy) (victim-status v8 healthy) (victim-status v9 healthy) (victim-status v10 healthy)))
 )
