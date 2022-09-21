(define (domain triangle_tireworld_mdp)
(:requirements :typing)
(:types location)
(:predicates
    (goal-reward-received )
    (spare-in ?p0 - location)
    (hasspare )
    (road ?p0 - location ?p1 - location)
    (vehicle-at ?p0 - location)
    (goal-location ?p0 - location)
    (not-flattire ))
(:action noop
    :parameters ()
    :precondition (and)
    :effect (and))
(:action changetire
    :parameters ()
    :precondition (and)
    :effect (and))
(:action move-car
    :parameters (?p0 - location ?p1 - location)
    :precondition (and)
    :effect (and))
(:action loadtire
    :parameters (?p0 - location)
    :precondition (and)
    :effect (and)))
; <magic_json> {"rddl": true}

