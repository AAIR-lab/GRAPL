(define (problem problem_0)
(:domain sysadmin_mdp)
(:objects
    c1 - computer
    c2 - computer
    c3 - computer
    c4 - computer
    c5 - computer
    c6 - computer)
(:init
    (running c1)
    (running c2)
    (running c3)
    (running c4)
    (running c5)
    (running c6)
    (CONNECTED c1 c2)
    (CONNECTED c1 c3)
    (CONNECTED c2 c4)
    (CONNECTED c2 c6)
    (CONNECTED c3 c2)
    (CONNECTED c4 c2)
    (CONNECTED c4 c6)
    (CONNECTED c5 c3)
    (CONNECTED c5 c4)
    (CONNECTED c6 c5))
(:goal (and)))
; <magic_json> {"rddl": true}

