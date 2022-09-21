(define (domain academic_advising_mdp)
(:requirements :typing)
(:types course)
(:predicates
    (taken ?p0 - course)
    (passed ?p0 - course)
    (PROGRAM_REQUIREMENT ?p0 - course)
    (PREREQ ?p0 - course ?p1 - course))
(:action noop
    :parameters ()
    :precondition (and)
    :effect (and))
(:action takeCourse
    :parameters (?p0 - course)
    :precondition (and)
    :effect (and)))
; <magic_json> {"rddl": true}

