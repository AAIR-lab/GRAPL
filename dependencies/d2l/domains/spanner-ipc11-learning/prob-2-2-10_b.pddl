(define (problem prob)
 (:domain spanner)
 (:objects 
     bob - man
 spanner1 spanner2 - spanner
     nut1 nut2 - nut
     location1 location2 location3 location4 location5 location6 location7 location8 location9 location10 - location
     shed gate - location
    )
 (:init 
    (at bob shed)
    (at spanner1 location1)
    (useable spanner1)
    (at spanner2 location7)
    (useable spanner2)
    (loose nut1)
    (at nut1 gate)
    (loose nut2)
    (at nut2 gate)
    (link shed location1)
    (link location10 gate)
    (link location1 location2)
    (link location2 location3)
    (link location3 location4)
    (link location4 location5)
    (link location5 location6)
    (link location6 location7)
    (link location7 location8)
    (link location8 location9)
    (link location9 location10)
)
 (:goal
  (and
   (tightened nut1)
   (tightened nut2)
)))
