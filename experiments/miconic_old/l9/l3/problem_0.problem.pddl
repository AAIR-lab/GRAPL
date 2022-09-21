


(define (problem mixed-f4-p6-u0-v0-d0-a0-n0-A0-B0-N0-F0)
   (:domain miconic)
   (:objects p0 p1 p2 p3 p4 p5 - passenger
             f0 f1 f2 f3 - floor)


(:init
(above f0 f1)
(above f0 f2)
(above f0 f3)

(above f1 f2)
(above f1 f3)

(above f2 f3)



(origin p0 f2)
(destin p0 f3)

(origin p1 f3)
(destin p1 f0)

(origin p2 f3)
(destin p2 f2)

(origin p3 f1)
(destin p3 f0)

(origin p4 f3)
(destin p4 f0)

(origin p5 f3)
(destin p5 f0)






(lift-at f0)
)


(:goal (and
(served p0)
(served p1)
(served p2)
(served p3)
(served p4)
(served p5)
))
)


; <magic_json> {"min_floors": 4, "max_floors": 4, "min_passengers": 6, "max_passengers": 6, "floors": 4, "passengers": 6, "bin_params": ["floors", "passengers"]}

