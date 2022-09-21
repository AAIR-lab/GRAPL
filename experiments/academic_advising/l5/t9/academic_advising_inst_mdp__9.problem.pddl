(define (problem academic_advising_inst_mdp__9)
(:domain academic_advising_mdp)
(:objects
    CS11 - course
    CS12 - course
    CS13 - course
    CS14 - course
    CS15 - course
    CS16 - course
    CS21 - course
    CS22 - course
    CS23 - course
    CS24 - course
    CS25 - course
    CS26 - course
    CS31 - course
    CS32 - course
    CS33 - course
    CS34 - course
    CS35 - course
    CS36 - course
    CS41 - course
    CS42 - course
    CS43 - course
    CS44 - course
    CS45 - course
    CS46 - course
    CS51 - course
    CS52 - course
    CS53 - course
    CS54 - course
    CS55 - course
    CS56 - course)
(:init
    (PROGRAM_REQUIREMENT CS12)
    (PROGRAM_REQUIREMENT CS15)
    (PROGRAM_REQUIREMENT CS16)
    (PREREQ CS13 CS21)
    (PREREQ CS12 CS22)
    (PREREQ CS16 CS22)
    (PREREQ CS21 CS23)
    (PREREQ CS23 CS24)
    (PREREQ CS11 CS24)
    (PROGRAM_REQUIREMENT CS24)
    (PREREQ CS11 CS25)
    (PREREQ CS12 CS26)
    (PROGRAM_REQUIREMENT CS26)
    (PREREQ CS25 CS31)
    (PREREQ CS21 CS31)
    (PREREQ CS23 CS32)
    (PREREQ CS13 CS32)
    (PREREQ CS32 CS33)
    (PREREQ CS12 CS33)
    (PREREQ CS24 CS33)
    (PROGRAM_REQUIREMENT CS33)
    (PREREQ CS13 CS34)
    (PREREQ CS23 CS34)
    (PROGRAM_REQUIREMENT CS34)
    (PREREQ CS32 CS35)
    (PREREQ CS33 CS36)
    (PREREQ CS26 CS36)
    (PREREQ CS14 CS36)
    (PREREQ CS16 CS41)
    (PREREQ CS24 CS41)
    (PREREQ CS11 CS41)
    (PROGRAM_REQUIREMENT CS41)
    (PREREQ CS32 CS42)
    (PREREQ CS35 CS43)
    (PREREQ CS14 CS43)
    (PREREQ CS14 CS44)
    (PREREQ CS43 CS44)
    (PREREQ CS22 CS45)
    (PREREQ CS13 CS45)
    (PREREQ CS22 CS46)
    (PREREQ CS33 CS46)
    (PREREQ CS23 CS46)
    (PROGRAM_REQUIREMENT CS46)
    (PREREQ CS12 CS51)
    (PREREQ CS34 CS51)
    (PROGRAM_REQUIREMENT CS51)
    (PREREQ CS41 CS52)
    (PREREQ CS41 CS53)
    (PREREQ CS53 CS54)
    (PREREQ CS44 CS54)
    (PREREQ CS24 CS55)
    (PREREQ CS41 CS55)
    (PREREQ CS22 CS56)
    (PREREQ CS45 CS56)
    (PREREQ CS54 CS56)
    (PROGRAM_REQUIREMENT CS56))
(:goal (and)))
; <magic_json> {"rddl": true}

