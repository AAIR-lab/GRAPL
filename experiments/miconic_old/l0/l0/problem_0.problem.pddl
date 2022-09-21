


(define (problem mixed-f3-p1-u0-v0-d0-a0-n0-A0-B0-N0-F0)
   (:domain miconic)
   (:objects p0 - passenger
             f0 f1 f2 - floor)


(:init
(above f0 f1)
(above f0 f2)

(above f1 f2)



(origin p0 f0)
(destin p0 f1)






(lift-at f0)
)


(:goal (and
(served p0)
))
)


; <magic_json> {"min_floors": 3, "max_floors": 3, "min_passengers": 1, "max_passengers": 1, "floors": 3, "passengers": 1, "bin_params": ["floors", "passengers"]}

