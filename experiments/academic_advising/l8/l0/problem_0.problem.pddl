(define (problem problem_0)
(:domain academic_advising_mdp)
(:objects
    CS11 - course
    CS12 - course
    CS21 - course
    CS22 - course)
(:init
    (PROGRAM_REQUIREMENT CS11)
    (PROGRAM_REQUIREMENT CS12)
    (PREREQ CS12 CS21)
    (PREREQ CS11 CS21)
    (PROGRAM_REQUIREMENT CS21)
    (PREREQ CS21 CS22)
    (PREREQ CS11 CS22))
(:goal (and)))
; <magic_json> {"rddl": true}

