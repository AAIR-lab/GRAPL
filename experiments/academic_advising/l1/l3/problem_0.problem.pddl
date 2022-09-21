(define (problem problem_0)
(:domain academic_advising_mdp)
(:objects
    CS11 - course
    CS12 - course
    CS13 - course
    CS14 - course
    CS21 - course
    CS22 - course
    CS23 - course
    CS24 - course
    CS31 - course
    CS32 - course
    CS33 - course
    CS34 - course)
(:init
    (PROGRAM_REQUIREMENT CS11)
    (PROGRAM_REQUIREMENT CS12)
    (PROGRAM_REQUIREMENT CS13)
    (PROGRAM_REQUIREMENT CS14)
    (PREREQ CS11 CS21)
    (PREREQ CS13 CS21)
    (PROGRAM_REQUIREMENT CS21)
    (PREREQ CS13 CS22)
    (PROGRAM_REQUIREMENT CS22)
    (PREREQ CS13 CS23)
    (PROGRAM_REQUIREMENT CS23)
    (PREREQ CS13 CS24)
    (PROGRAM_REQUIREMENT CS24)
    (PREREQ CS23 CS31)
    (PROGRAM_REQUIREMENT CS31)
    (PREREQ CS12 CS32)
    (PREREQ CS14 CS32)
    (PROGRAM_REQUIREMENT CS32)
    (PREREQ CS23 CS33)
    (PREREQ CS22 CS33)
    (PROGRAM_REQUIREMENT CS33)
    (PREREQ CS33 CS34)
    (PROGRAM_REQUIREMENT CS34))
(:goal (and)))
; <magic_json> {"rddl": true}

