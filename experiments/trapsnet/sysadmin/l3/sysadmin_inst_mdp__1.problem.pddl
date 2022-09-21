(define (problem sysadmin_inst_mdp__1)
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
    (CONNECTED c1 c4)
    (CONNECTED c1 c9)
    (CONNECTED c2 c8)
    (CONNECTED c3 c4)
    (CONNECTED c3 c9)
    (CONNECTED c4 c5)
    (CONNECTED c5 c7)
    (CONNECTED c6 c4)
    (CONNECTED c6 c8)
    (CONNECTED c7 c9)
    (CONNECTED c8 c6)
    (CONNECTED c8 c10)
    (CONNECTED c9 c6)
    (CONNECTED c10 c2))
(:goal (and)))
; <magic_json> {"rddl": true}

