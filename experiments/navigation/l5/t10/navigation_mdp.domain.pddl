(define (domain navigation_mdp)
(:requirements :typing)
(:types ypos xpos)
(:predicates
    (MAX-XPOS ?p0 - xpos)
    (MAX-YPOS ?p0 - ypos)
    (GOAL ?p0 - xpos ?p1 - ypos)
    (SOUTH ?p0 - ypos ?p1 - ypos)
    (NORTH ?p0 - ypos ?p1 - ypos)
    (robot-at ?p0 - xpos ?p1 - ypos)
    (EAST ?p0 - xpos ?p1 - xpos)
    (WEST ?p0 - xpos ?p1 - xpos)
    (MIN-YPOS ?p0 - ypos)
    (MIN-XPOS ?p0 - xpos))
(:action noop
    :parameters ()
    :precondition (and)
    :effect (and))
(:action move-east
    :parameters ()
    :precondition (and)
    :effect (and))
(:action move-north
    :parameters ()
    :precondition (and)
    :effect (and))
(:action move-south
    :parameters ()
    :precondition (and)
    :effect (and))
(:action move-west
    :parameters ()
    :precondition (and)
    :effect (and)))
; <magic_json> {"rddl": true}

