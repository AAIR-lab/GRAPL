(define (problem debug)
   (:domain miconic)
   (:objects p0 p1
             f0 f1 f2)


(:init
(passenger p0)
(passenger p1)
(floor f0)
(floor f1)
(floor f2)

(above f0 f1)
(above f0 f2)

(above f1 f2)

(origin p0 f2)
(destin p0 f1)

(origin p1 f1)
(destin p1 f0)

(lift-at f0)
)


(:goal (and 
(served p0)
(served p1)
))
)


