(define (domain wildfire_2_mdp)
(:requirements :typing)
(:types location)
(:predicates
    (burning ?p0 - location)
    (out-of-fuel ?p0 - location)
    (NEIGHBOR ?p0 - location ?p1 - location)
    (TARGET ?p0 - location))
(:action noop
    :parameters ()
    :precondition (and)
    :effect (and))
(:action put-out
    :parameters (?p0 - location)
    :precondition (and)
    :effect (and))
(:action cut-out
    :parameters (?p0 - location)
    :precondition (and)
    :effect (and)))
; <magic_json> {"rddl": true}

