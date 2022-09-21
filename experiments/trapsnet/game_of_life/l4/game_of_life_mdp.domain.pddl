(define (domain game_of_life_mdp)
(:requirements :typing)
(:types x_pos y_pos)
(:predicates
    (alive ?p0 - x_pos ?p1 - y_pos)
    (NEIGHBOR ?p0 - x_pos ?p1 - y_pos ?p2 - x_pos ?p3 - y_pos))
(:action noop
    :parameters ()
    :precondition (and)
    :effect (and))
(:action set
    :parameters (?p0 - x_pos ?p1 - y_pos)
    :precondition (and)
    :effect (and)))
; <magic_json> {"rddl": true}

