(define (domain sysadmin_mdp)
(:requirements :typing)
(:types computer)
(:predicates
    (CONNECTED ?p0 - computer ?p1 - computer)
    (running ?p0 - computer))
(:action noop
    :parameters ()
    :precondition (and)
    :effect (and))
(:action reboot
    :parameters (?p0 - computer)
    :precondition (and)
    :effect (and)))
; <magic_json> {"rddl": true}

