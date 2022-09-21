(define (problem academic_advising_inst_mdp__1)
(:domain academic_advising_mdp)
(:objects
    CS11 - course
    CS12 - course
    CS21 - course
    CS22 - course
    CS31 - course
    CS32 - course
    CS41 - course
    CS42 - course
    CS51 - course
    CS52 - course)
(:init
    (PREREQ CS12 CS21)
    (PREREQ CS11 CS21)
    (PROGRAM_REQUIREMENT CS21)
    (PREREQ CS12 CS22)
    (PREREQ CS21 CS22)
    (PROGRAM_REQUIREMENT CS22)
    (PREREQ CS21 CS31)
    (PREREQ CS11 CS31)
    (PREREQ CS22 CS32)
    (PREREQ CS11 CS32)
    (PREREQ CS11 CS41)
    (PREREQ CS22 CS41)
    (PROGRAM_REQUIREMENT CS41)
    (PREREQ CS11 CS42)
    (PREREQ CS31 CS42)
    (PREREQ CS41 CS51)
    (PREREQ CS12 CS51)
    (PREREQ CS31 CS52)
    (PREREQ CS22 CS52))
(:goal (and)))
; <magic_json> {"rddl": true}

