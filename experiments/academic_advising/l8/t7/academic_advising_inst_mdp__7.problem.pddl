(define (problem academic_advising_inst_mdp__7)
(:domain academic_advising_mdp)
(:objects
    CS11 - course
    CS12 - course
    CS13 - course
    CS14 - course
    CS15 - course
    CS21 - course
    CS22 - course
    CS23 - course
    CS24 - course
    CS25 - course
    CS31 - course
    CS32 - course
    CS33 - course
    CS34 - course
    CS35 - course
    CS41 - course
    CS42 - course
    CS43 - course
    CS44 - course
    CS45 - course
    CS51 - course
    CS52 - course
    CS53 - course
    CS54 - course
    CS55 - course)
(:init
    (PROGRAM_REQUIREMENT CS12)
    (PROGRAM_REQUIREMENT CS13)
    (PREREQ CS12 CS21)
    (PREREQ CS13 CS21)
    (PREREQ CS15 CS21)
    (PREREQ CS11 CS22)
    (PREREQ CS14 CS23)
    (PREREQ CS15 CS23)
    (PREREQ CS21 CS24)
    (PREREQ CS14 CS25)
    (PREREQ CS12 CS25)
    (PROGRAM_REQUIREMENT CS25)
    (PREREQ CS12 CS31)
    (PREREQ CS24 CS31)
    (PREREQ CS14 CS31)
    (PROGRAM_REQUIREMENT CS31)
    (PREREQ CS22 CS32)
    (PREREQ CS22 CS33)
    (PREREQ CS25 CS33)
    (PREREQ CS32 CS33)
    (PREREQ CS22 CS34)
    (PREREQ CS11 CS34)
    (PROGRAM_REQUIREMENT CS34)
    (PREREQ CS12 CS35)
    (PREREQ CS34 CS41)
    (PREREQ CS35 CS41)
    (PROGRAM_REQUIREMENT CS41)
    (PREREQ CS41 CS42)
    (PREREQ CS34 CS42)
    (PREREQ CS13 CS42)
    (PROGRAM_REQUIREMENT CS42)
    (PREREQ CS12 CS43)
    (PREREQ CS22 CS43)
    (PREREQ CS21 CS44)
    (PREREQ CS31 CS45)
    (PREREQ CS43 CS45)
    (PREREQ CS41 CS51)
    (PREREQ CS43 CS51)
    (PREREQ CS22 CS52)
    (PREREQ CS12 CS52)
    (PREREQ CS33 CS52)
    (PROGRAM_REQUIREMENT CS52)
    (PREREQ CS25 CS53)
    (PREREQ CS33 CS54)
    (PREREQ CS13 CS55)
    (PREREQ CS41 CS55))
(:goal (and)))
; <magic_json> {"rddl": true}

