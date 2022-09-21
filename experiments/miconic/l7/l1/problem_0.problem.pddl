


(define (problem mixed-f3-p2-u0-v0-d0-a0-n0-A0-B0-N0-F0)
   (:domain miconic)
   (:objects p0 p1 - passenger
             f0 f1 f2 - floor)


(:init
(above f0 f1)
(above f0 f2)

(above f1 f2)



(origin p0 f0)
(destin p0 f1)

(origin p1 f2)
(destin p1 f1)






(lift-at f0)
)


(:goal (and
(served p0)
(served p1)
))
)


; <magic_json> {"min_floors": 3, "max_floors": 3, "min_passengers": 2, "max_passengers": 2, "floors": 3, "passengers": 2, "bin_params": ["floors", "passengers"]}

