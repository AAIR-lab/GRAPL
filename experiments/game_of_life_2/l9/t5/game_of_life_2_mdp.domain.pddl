(define (domain game_of_life_2_mdp)
(:requirements :typing)
(:types location)
(:predicates
    (alive ?p0 - location)
    (NEIGHBOR ?p0 - location ?p1 - location))
(:action noop
    :parameters ()
    :precondition (and)
    :effect (and))
(:action set
    :parameters (?p0 - location)
    :precondition (and)
    :effect (and)))
; <magic_json> {"rddl": true}

