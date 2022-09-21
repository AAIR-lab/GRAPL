(define (problem problem_0)
(:domain academic_advising_mdp)
(:objects
    CS11 - course
    CS12 - course
    CS13 - course
    CS21 - course
    CS22 - course
    CS23 - course
    CS31 - course
    CS32 - course
    CS33 - course)
(:init
    (PROGRAM_REQUIREMENT CS12)
    (PREREQ CS11 CS21)
    (PREREQ CS13 CS21)
    (PREREQ CS12 CS21)
    (PROGRAM_REQUIREMENT CS21)
    (PREREQ CS13 CS22)
    (PREREQ CS12 CS22)
    (PREREQ CS21 CS22)
    (PREREQ CS11 CS22)
    (PROGRAM_REQUIREMENT CS22)
    (PREREQ CS21 CS23)
    (PREREQ CS12 CS23)
    (PREREQ CS22 CS31)
    (PREREQ CS21 CS31)
    (PREREQ CS23 CS31)
    (PREREQ CS23 CS32)
    (PREREQ CS12 CS32)
    (PREREQ CS11 CS32)
    (PROGRAM_REQUIREMENT CS32)
    (PREREQ CS13 CS33)
    (PREREQ CS11 CS33)
    (PROGRAM_REQUIREMENT CS33))
(:goal (and)))
; <magic_json> {"rddl": true}

