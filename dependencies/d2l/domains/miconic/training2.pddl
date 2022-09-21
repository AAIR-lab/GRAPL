


(define (problem training)
   (:domain miconic)
   (:objects p0 p1
             f0 f1 )


(:init
(passenger p0)
(passenger p1)

(floor f0)
(floor f1)

(above f0 f1)

(destin p0 f1)
(destin p1 f1)

(lift-at f1)
(boarded p0)
(boarded p1)
)


(:goal (and 
(served p0)
(served p1)
))
)


