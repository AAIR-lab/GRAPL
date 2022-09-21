(define (problem problem_0)
(:domain sysadmin_mdp)
(:objects
    c1 - computer
    c2 - computer
    c3 - computer
    c4 - computer
    c5 - computer
    c6 - computer
    c7 - computer
    c8 - computer
    c9 - computer
    c10 - computer
    c11 - computer
    c12 - computer
    c13 - computer
    c14 - computer
    c15 - computer
    c16 - computer
    c17 - computer
    c18 - computer
    c19 - computer
    c20 - computer)
(:init
    (running c1)
    (running c2)
    (running c3)
    (running c4)
    (running c5)
    (running c6)
    (running c7)
    (running c8)
    (running c9)
    (running c10)
    (running c11)
    (running c12)
    (running c13)
    (running c14)
    (running c15)
    (running c16)
    (running c17)
    (running c18)
    (running c19)
    (running c20)
    (CONNECTED c1 c16)
    (CONNECTED c1 c4)
    (CONNECTED c2 c4)
    (CONNECTED c2 c11)
    (CONNECTED c3 c1)
    (CONNECTED c3 c11)
    (CONNECTED c4 c1)
    (CONNECTED c4 c2)
    (CONNECTED c5 c16)
    (CONNECTED c5 c11)
    (CONNECTED c6 c20)
    (CONNECTED c6 c11)
    (CONNECTED c7 c17)
    (CONNECTED c7 c19)
    (CONNECTED c8 c17)
    (CONNECTED c8 c5)
    (CONNECTED c9 c19)
    (CONNECTED c10 c16)
    (CONNECTED c10 c12)
    (CONNECTED c11 c14)
    (CONNECTED c11 c15)
    (CONNECTED c12 c17)
    (CONNECTED c12 c13)
    (CONNECTED c13 c8)
    (CONNECTED c13 c12)
    (CONNECTED c14 c4)
    (CONNECTED c15 c10)
    (CONNECTED c15 c14)
    (CONNECTED c16 c13)
    (CONNECTED c17 c16)
    (CONNECTED c17 c18)
    (CONNECTED c18 c2)
    (CONNECTED c18 c15)
    (CONNECTED c19 c5)
    (CONNECTED c20 c18)
    (CONNECTED c20 c2))
(:goal (and)))
; <magic_json> {"rddl": true}

