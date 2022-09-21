(define (problem sysadmin_inst_mdp__2)
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
    c10 - computer)
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
    (CONNECTED c1 c5)
    (CONNECTED c1 c10)
    (CONNECTED c2 c1)
    (CONNECTED c2 c7)
    (CONNECTED c2 c8)
    (CONNECTED c3 c4)
    (CONNECTED c3 c7)
    (CONNECTED c4 c3)
    (CONNECTED c4 c5)
    (CONNECTED c4 c6)
    (CONNECTED c5 c1)
    (CONNECTED c5 c2)
    (CONNECTED c5 c8)
    (CONNECTED c6 c3)
    (CONNECTED c6 c7)
    (CONNECTED c6 c10)
    (CONNECTED c7 c2)
    (CONNECTED c7 c8)
    (CONNECTED c7 c9)
    (CONNECTED c8 c1)
    (CONNECTED c8 c4)
    (CONNECTED c8 c5)
    (CONNECTED c9 c2)
    (CONNECTED c9 c5)
    (CONNECTED c9 c6)
    (CONNECTED c10 c2)
    (CONNECTED c10 c4)
    (CONNECTED c10 c8))
(:goal (and)))
; <magic_json> {"rddl": true}

