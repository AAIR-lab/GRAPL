(define (domain wildfire_mdp)
(:requirements :typing)
(:types y_pos x_pos)
(:predicates
    (burning ?p0 - x_pos ?p1 - y_pos)
    (out-of-fuel ?p0 - x_pos ?p1 - y_pos)
    (NEIGHBOR ?p0 - x_pos ?p1 - y_pos ?p2 - x_pos ?p3 - y_pos)
    (TARGET ?p0 - x_pos ?p1 - y_pos))
(:action noop
    :parameters ()
    :precondition (and)
    :effect (and))
(:action put-out
    :parameters (?p0 - x_pos ?p1 - y_pos)
    :precondition (and)
    :effect (and))
(:action cut-out
    :parameters (?p0 - x_pos ?p1 - y_pos)
    :precondition (and)
    :effect (and)))
; <magic_json> {"rddl": true}

