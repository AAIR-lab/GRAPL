(define (problem FR_15_20)
 (:domain first-response)
 (:objects  l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15  - location
	    f1 f2 f3 f4 f5 f6 f7 f8 - fire_unit
	    v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 - victim
	    m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 - medical_unit
)
 (:init 
	;;strategic locations
     (hospital l14)
     (hospital l5)
     (hospital l14)
     (water-at l2)
     (water-at l2)
     (water-at l15)
     (water-at l3)
     (water-at l7)
     (water-at l15)
	;;disaster info
     (fire l3)
     (victim-at v1 l12)
     (victim-status v1 dying)
     (fire l2)
     (victim-at v2 l4)
     (victim-status v2 dying)
     (fire l9)
     (victim-at v3 l1)
     (victim-status v3 hurt)
     (fire l13)
     (victim-at v4 l1)
     (victim-status v4 dying)
     (fire l9)
     (victim-at v5 l1)
     (victim-status v5 dying)
     (fire l12)
     (victim-at v6 l11)
     (victim-status v6 hurt)
     (fire l11)
     (victim-at v7 l13)
     (victim-status v7 dying)
     (fire l11)
     (victim-at v8 l13)
     (victim-status v8 dying)
     (fire l7)
     (victim-at v9 l3)
     (victim-status v9 dying)
     (fire l6)
     (victim-at v10 l2)
     (victim-status v10 hurt)
     (fire l5)
     (victim-at v11 l3)
     (victim-status v11 dying)
     (fire l2)
     (victim-at v12 l3)
     (victim-status v12 hurt)
     (fire l1)
     (victim-at v13 l9)
     (victim-status v13 dying)
     (fire l2)
     (victim-at v14 l12)
     (victim-status v14 dying)
     (fire l9)
     (victim-at v15 l15)
     (victim-status v15 hurt)
     (fire l7)
     (victim-at v16 l4)
     (victim-status v16 hurt)
     (fire l9)
     (victim-at v17 l6)
     (victim-status v17 dying)
     (fire l5)
     (victim-at v18 l12)
     (victim-status v18 hurt)
     (fire l13)
     (victim-at v19 l1)
     (victim-status v19 hurt)
     (fire l6)
     (victim-at v20 l5)
     (victim-status v20 hurt)
	;;map info
	(adjacent l1 l1)
	(adjacent l2 l2)
	(adjacent l3 l3)
	(adjacent l4 l4)
	(adjacent l5 l5)
	(adjacent l6 l6)
	(adjacent l7 l7)
	(adjacent l8 l8)
	(adjacent l9 l9)
	(adjacent l10 l10)
	(adjacent l11 l11)
	(adjacent l12 l12)
	(adjacent l13 l13)
	(adjacent l14 l14)
	(adjacent l15 l15)
   (adjacent l1 l1)
   (adjacent l1 l1)
   (adjacent l1 l2)
   (adjacent l2 l1)
   (adjacent l1 l3)
   (adjacent l3 l1)
   (adjacent l1 l4)
   (adjacent l4 l1)
   (adjacent l1 l5)
   (adjacent l5 l1)
   (adjacent l1 l6)
   (adjacent l6 l1)
   (adjacent l1 l7)
   (adjacent l7 l1)
   (adjacent l1 l8)
   (adjacent l8 l1)
   (adjacent l1 l9)
   (adjacent l9 l1)
   (adjacent l1 l10)
   (adjacent l10 l1)
   (adjacent l1 l11)
   (adjacent l11 l1)
   (adjacent l1 l12)
   (adjacent l12 l1)
   (adjacent l1 l13)
   (adjacent l13 l1)
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
   (adjacent l2 l6)
   (adjacent l6 l2)
   (adjacent l2 l7)
   (adjacent l7 l2)
   (adjacent l2 l8)
   (adjacent l8 l2)
   (adjacent l2 l9)
   (adjacent l9 l2)
   (adjacent l2 l10)
   (adjacent l10 l2)
   (adjacent l2 l11)
   (adjacent l11 l2)
   (adjacent l2 l12)
   (adjacent l12 l2)
   (adjacent l2 l13)
   (adjacent l13 l2)
   (adjacent l3 l1)
   (adjacent l1 l3)
   (adjacent l3 l2)
   (adjacent l2 l3)
   (adjacent l3 l3)
   (adjacent l3 l3)
   (adjacent l3 l4)
   (adjacent l4 l3)
   (adjacent l3 l5)
   (adjacent l5 l3)
   (adjacent l3 l6)
   (adjacent l6 l3)
   (adjacent l4 l1)
   (adjacent l1 l4)
   (adjacent l4 l2)
   (adjacent l2 l4)
   (adjacent l4 l3)
   (adjacent l3 l4)
   (adjacent l4 l4)
   (adjacent l4 l4)
   (adjacent l4 l5)
   (adjacent l5 l4)
   (adjacent l4 l6)
   (adjacent l6 l4)
   (adjacent l4 l7)
   (adjacent l7 l4)
   (adjacent l4 l8)
   (adjacent l8 l4)
   (adjacent l4 l9)
   (adjacent l9 l4)
   (adjacent l4 l10)
   (adjacent l10 l4)
   (adjacent l4 l11)
   (adjacent l11 l4)
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
   (adjacent l6 l1)
   (adjacent l1 l6)
   (adjacent l6 l2)
   (adjacent l2 l6)
   (adjacent l6 l3)
   (adjacent l3 l6)
   (adjacent l6 l4)
   (adjacent l4 l6)
   (adjacent l6 l5)
   (adjacent l5 l6)
   (adjacent l6 l6)
   (adjacent l6 l6)
   (adjacent l7 l1)
   (adjacent l1 l7)
   (adjacent l7 l2)
   (adjacent l2 l7)
   (adjacent l8 l1)
   (adjacent l1 l8)
   (adjacent l8 l2)
   (adjacent l2 l8)
   (adjacent l8 l3)
   (adjacent l3 l8)
   (adjacent l8 l4)
   (adjacent l4 l8)
   (adjacent l8 l5)
   (adjacent l5 l8)
   (adjacent l8 l6)
   (adjacent l6 l8)
   (adjacent l9 l1)
   (adjacent l1 l9)
   (adjacent l9 l2)
   (adjacent l2 l9)
   (adjacent l9 l3)
   (adjacent l3 l9)
   (adjacent l9 l4)
   (adjacent l4 l9)
   (adjacent l9 l5)
   (adjacent l5 l9)
   (adjacent l9 l6)
   (adjacent l6 l9)
   (adjacent l9 l7)
   (adjacent l7 l9)
   (adjacent l9 l8)
   (adjacent l8 l9)
   (adjacent l9 l9)
   (adjacent l9 l9)
   (adjacent l9 l10)
   (adjacent l10 l9)
   (adjacent l9 l11)
   (adjacent l11 l9)
   (adjacent l9 l12)
   (adjacent l12 l9)
   (adjacent l9 l13)
   (adjacent l13 l9)
   (adjacent l9 l14)
   (adjacent l14 l9)
   (adjacent l10 l1)
   (adjacent l1 l10)
   (adjacent l10 l2)
   (adjacent l2 l10)
   (adjacent l10 l3)
   (adjacent l3 l10)
   (adjacent l10 l4)
   (adjacent l4 l10)
   (adjacent l10 l5)
   (adjacent l5 l10)
   (adjacent l10 l6)
   (adjacent l6 l10)
   (adjacent l10 l7)
   (adjacent l7 l10)
   (adjacent l10 l8)
   (adjacent l8 l10)
   (adjacent l10 l9)
   (adjacent l9 l10)
   (adjacent l10 l10)
   (adjacent l10 l10)
   (adjacent l10 l11)
   (adjacent l11 l10)
   (adjacent l10 l12)
   (adjacent l12 l10)
   (adjacent l10 l13)
   (adjacent l13 l10)
   (adjacent l11 l1)
   (adjacent l1 l11)
   (adjacent l11 l2)
   (adjacent l2 l11)
   (adjacent l11 l3)
   (adjacent l3 l11)
   (adjacent l11 l4)
   (adjacent l4 l11)
   (adjacent l11 l5)
   (adjacent l5 l11)
   (adjacent l11 l6)
   (adjacent l6 l11)
   (adjacent l11 l7)
   (adjacent l7 l11)
   (adjacent l11 l8)
   (adjacent l8 l11)
   (adjacent l11 l9)
   (adjacent l9 l11)
   (adjacent l11 l10)
   (adjacent l10 l11)
   (adjacent l11 l11)
   (adjacent l11 l11)
   (adjacent l11 l12)
   (adjacent l12 l11)
   (adjacent l11 l13)
   (adjacent l13 l11)
   (adjacent l11 l14)
   (adjacent l14 l11)
   (adjacent l12 l1)
   (adjacent l1 l12)
   (adjacent l12 l2)
   (adjacent l2 l12)
   (adjacent l12 l3)
   (adjacent l3 l12)
   (adjacent l12 l4)
   (adjacent l4 l12)
   (adjacent l12 l5)
   (adjacent l5 l12)
   (adjacent l12 l6)
   (adjacent l6 l12)
   (adjacent l12 l7)
   (adjacent l7 l12)
   (adjacent l12 l8)
   (adjacent l8 l12)
   (adjacent l13 l1)
   (adjacent l1 l13)
   (adjacent l13 l2)
   (adjacent l2 l13)
   (adjacent l13 l3)
   (adjacent l3 l13)
   (adjacent l13 l4)
   (adjacent l4 l13)
   (adjacent l14 l1)
   (adjacent l1 l14)
   (adjacent l14 l2)
   (adjacent l2 l14)
   (adjacent l14 l3)
   (adjacent l3 l14)
   (adjacent l14 l4)
   (adjacent l4 l14)
   (adjacent l14 l5)
   (adjacent l5 l14)
   (adjacent l14 l6)
   (adjacent l6 l14)
   (adjacent l14 l7)
   (adjacent l7 l14)
   (adjacent l15 l1)
   (adjacent l1 l15)
   (adjacent l15 l2)
   (adjacent l2 l15)
   (adjacent l15 l3)
   (adjacent l3 l15)
   (adjacent l15 l4)
   (adjacent l4 l15)
	(fire-unit-at f1 l15)
	(fire-unit-at f2 l2)
	(fire-unit-at f3 l3)
	(fire-unit-at f4 l14)
	(fire-unit-at f5 l2)
	(fire-unit-at f6 l2)
	(fire-unit-at f7 l5)
	(fire-unit-at f8 l6)
	(medical-unit-at m1 l1)
	(medical-unit-at m2 l6)
	(medical-unit-at m3 l5)
	(medical-unit-at m4 l10)
	(medical-unit-at m5 l8)
	(medical-unit-at m6 l14)
	(medical-unit-at m7 l6)
	(medical-unit-at m8 l10)
	(medical-unit-at m9 l13)
	(medical-unit-at m10 l7)
	(medical-unit-at m11 l15)
	)
 (:goal (and  (nfire l3) (nfire l2) (nfire l9) (nfire l13) (nfire l9) (nfire l12) (nfire l11) (nfire l11) (nfire l7) (nfire l6) (nfire l5) (nfire l2) (nfire l1) (nfire l2) (nfire l9) (nfire l7) (nfire l9) (nfire l5) (nfire l13) (nfire l6)  (victim-status v1 healthy) (victim-status v2 healthy) (victim-status v3 healthy) (victim-status v4 healthy) (victim-status v5 healthy) (victim-status v6 healthy) (victim-status v7 healthy) (victim-status v8 healthy) (victim-status v9 healthy) (victim-status v10 healthy) (victim-status v11 healthy) (victim-status v12 healthy) (victim-status v13 healthy) (victim-status v14 healthy) (victim-status v15 healthy) (victim-status v16 healthy) (victim-status v17 healthy) (victim-status v18 healthy) (victim-status v19 healthy) (victim-status v20 healthy)))
 )
