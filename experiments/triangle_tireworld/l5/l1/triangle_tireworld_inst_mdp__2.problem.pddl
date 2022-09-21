(define (problem triangle_tireworld_inst_mdp__2)
(:domain triangle_tireworld_mdp)
(:objects
    la1a1 - location
    la1a2 - location
    la1a3 - location
    la2a1 - location
    la2a2 - location
    la3a1 - location)
(:init
    (vehicle-at la1a1)
    (spare-in la2a1)
    (spare-in la2a2)
    (spare-in la3a1)
    (spare-in la3a1)
    (not-flattire )
    (road la1a1 la1a2)
    (road la1a2 la1a3)
    (road la1a1 la2a1)
    (road la1a2 la2a2)
    (road la2a1 la1a2)
    (road la2a2 la1a3)
    (road la2a1 la3a1)
    (road la3a1 la2a2)
    (goal-location la1a3))
(:goal (and)))
; <magic_json> {"rddl": true}

