(define (problem academic_advising_inst_mdp__3)
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
    CS33 - course
    CS41 - course
    CS42 - course
    CS43 - course
    CS51 - course
    CS52 - course
    CS53 - course)
(:init
    (PROGRAM_REQUIREMENT CS12)
    (PROGRAM_REQUIREMENT CS13)
    (PREREQ CS12 CS21)
    (PREREQ CS11 CS22)
    (PREREQ CS13 CS22)
    (PREREQ CS22 CS23)
    (PREREQ CS11 CS23)
    (PREREQ CS11 CS31)
    (PREREQ CS12 CS31)
    (PROGRAM_REQUIREMENT CS31)
    (PREREQ CS11 CS32)
    (PREREQ CS23 CS32)
    (PREREQ CS22 CS32)
    (PREREQ CS13 CS33)
    (PREREQ CS11 CS33)
    (PREREQ CS12 CS41)
    (PROGRAM_REQUIREMENT CS41)
    (PREREQ CS22 CS42)
    (PREREQ CS12 CS42)
    (PREREQ CS12 CS43)
    (PREREQ CS22 CS43)
    (PREREQ CS11 CS43)
    (PREREQ CS33 CS51)
    (PREREQ CS31 CS51)
    (PREREQ CS42 CS52)
    (PREREQ CS52 CS53)
    (PREREQ CS33 CS53))
(:goal (and)))
; <magic_json> {"rddl": true}

