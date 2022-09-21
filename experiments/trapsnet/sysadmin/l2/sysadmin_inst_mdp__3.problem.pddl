(define (problem sysadmin_inst_mdp__3)
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
    (CONNECTED c1 c15)
    (CONNECTED c2 c16)
    (CONNECTED c2 c15)
    (CONNECTED c3 c16)
    (CONNECTED c3 c6)
    (CONNECTED c4 c16)
    (CONNECTED c4 c1)
    (CONNECTED c5 c16)
    (CONNECTED c5 c20)
    (CONNECTED c6 c3)
    (CONNECTED c7 c4)
    (CONNECTED c7 c11)
    (CONNECTED c8 c9)
    (CONNECTED c8 c12)
    (CONNECTED c9 c1)
    (CONNECTED c9 c6)
    (CONNECTED c10 c4)
    (CONNECTED c10 c6)
    (CONNECTED c11 c17)
    (CONNECTED c11 c19)
    (CONNECTED c12 c19)
    (CONNECTED c12 c11)
    (CONNECTED c13 c1)
    (CONNECTED c13 c9)
    (CONNECTED c14 c2)
    (CONNECTED c14 c7)
    (CONNECTED c15 c17)
    (CONNECTED c15 c6)
    (CONNECTED c16 c8)
    (CONNECTED c16 c15)
    (CONNECTED c17 c2)
    (CONNECTED c17 c10)
    (CONNECTED c18 c1)
    (CONNECTED c18 c3)
    (CONNECTED c19 c1)
    (CONNECTED c19 c13)
    (CONNECTED c20 c6)
    (CONNECTED c20 c9))
(:goal (and)))
; <magic_json> {"rddl": true}

