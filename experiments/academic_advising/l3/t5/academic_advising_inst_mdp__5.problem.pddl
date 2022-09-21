(define (problem academic_advising_inst_mdp__5)
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
    CS34 - course
    CS41 - course
    CS42 - course
    CS43 - course
    CS44 - course
    CS51 - course
    CS52 - course
    CS53 - course
    CS54 - course)
(:init
    (PROGRAM_REQUIREMENT CS12)
    (PROGRAM_REQUIREMENT CS14)
    (PREREQ CS13 CS21)
    (PREREQ CS11 CS21)
    (PREREQ CS12 CS21)
    (PROGRAM_REQUIREMENT CS21)
    (PREREQ CS14 CS22)
    (PREREQ CS11 CS22)
    (PREREQ CS14 CS23)
    (PREREQ CS12 CS24)
    (PREREQ CS21 CS24)
    (PREREQ CS11 CS24)
    (PREREQ CS22 CS31)
    (PREREQ CS13 CS31)
    (PREREQ CS24 CS32)
    (PREREQ CS14 CS32)
    (PREREQ CS22 CS32)
    (PREREQ CS13 CS33)
    (PREREQ CS21 CS33)
    (PROGRAM_REQUIREMENT CS33)
    (PREREQ CS24 CS34)
    (PREREQ CS22 CS34)
    (PROGRAM_REQUIREMENT CS34)
    (PREREQ CS32 CS41)
    (PREREQ CS31 CS41)
    (PREREQ CS21 CS42)
    (PREREQ CS32 CS42)
    (PREREQ CS23 CS42)
    (PROGRAM_REQUIREMENT CS42)
    (PREREQ CS21 CS43)
    (PREREQ CS41 CS43)
    (PROGRAM_REQUIREMENT CS43)
    (PREREQ CS22 CS44)
    (PREREQ CS42 CS44)
    (PREREQ CS41 CS51)
    (PREREQ CS11 CS51)
    (PREREQ CS14 CS52)
    (PREREQ CS34 CS53)
    (PREREQ CS51 CS53)
    (PROGRAM_REQUIREMENT CS53)
    (PREREQ CS52 CS54))
(:goal (and)))
; <magic_json> {"rddl": true}
