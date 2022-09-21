(define (problem triangle_tireworld_inst_mdp__6)
(:domain triangle_tireworld_mdp)
(:objects
    la1a1 - location
    la1a2 - location
    la1a3 - location
    la1a4 - location
    la1a5 - location
    la1a6 - location
    la1a7 - location
    la2a1 - location
    la2a2 - location
    la2a3 - location
    la2a4 - location
    la2a5 - location
    la2a6 - location
    la3a1 - location
    la3a2 - location
    la3a3 - location
    la3a4 - location
    la3a5 - location
    la4a1 - location
    la4a2 - location
    la4a3 - location
    la4a4 - location
    la5a1 - location
    la5a2 - location
    la5a3 - location
    la6a1 - location
    la6a2 - location
    la7a1 - location)
(:init
    (vehicle-at la1a1)
    (spare-in la2a1)
    (spare-in la2a2)
    (spare-in la2a3)
    (spare-in la2a4)
    (spare-in la2a5)
    (spare-in la2a6)
    (spare-in la3a1)
    (spare-in la3a5)
    (spare-in la4a1)
    (spare-in la4a2)
    (spare-in la4a3)
    (spare-in la4a4)
    (spare-in la5a1)
    (spare-in la5a3)
    (spare-in la6a1)
    (spare-in la6a2)
    (spare-in la7a1)
    (spare-in la7a1)
    (not-flattire )
    (road la1a1 la1a2)
    (road la1a2 la1a3)
    (road la1a3 la1a4)
    (road la1a4 la1a5)
    (road la1a5 la1a6)
    (road la1a6 la1a7)
    (road la1a1 la2a1)
    (road la1a2 la2a2)
    (road la1a3 la2a3)
    (road la1a4 la2a4)
    (road la1a5 la2a5)
    (road la1a6 la2a6)
    (road la2a1 la1a2)
    (road la2a2 la1a3)
    (road la2a3 la1a4)
    (road la2a4 la1a5)
    (road la2a5 la1a6)
    (road la2a6 la1a7)
    (road la3a1 la3a2)
    (road la3a2 la3a3)
    (road la3a3 la3a4)
    (road la3a4 la3a5)
    (road la2a1 la3a1)
    (road la2a3 la3a3)
    (road la2a5 la3a5)
    (road la3a1 la2a2)
    (road la3a3 la2a4)
    (road la3a5 la2a6)
    (road la3a1 la4a1)
    (road la3a2 la4a2)
    (road la3a3 la4a3)
    (road la3a4 la4a4)
    (road la4a1 la3a2)
    (road la4a2 la3a3)
    (road la4a3 la3a4)
    (road la4a4 la3a5)
    (road la5a1 la5a2)
    (road la5a2 la5a3)
    (road la4a1 la5a1)
    (road la4a3 la5a3)
    (road la5a1 la4a2)
    (road la5a3 la4a4)
    (road la5a1 la6a1)
    (road la5a2 la6a2)
    (road la6a1 la5a2)
    (road la6a2 la5a3)
    (road la6a1 la7a1)
    (road la7a1 la6a2)
    (goal-location la1a7))
(:goal (and)))
; <magic_json> {"rddl": true}

