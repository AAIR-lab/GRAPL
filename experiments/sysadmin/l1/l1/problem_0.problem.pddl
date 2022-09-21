(define (problem problem_0)
(:domain sysadmin_mdp)
(:objects
    c1 - computer
    c2 - computer
    c3 - computer
    c4 - computer)
(:init
    (running c1)
    (running c2)
    (running c3)
    (running c4)
    (CONNECTED c1 c3)
    (CONNECTED c1 c4)
    (CONNECTED c2 c1)
    (CONNECTED c2 c4)
    (CONNECTED c3 c1)
    (CONNECTED c4 c1)
    (CONNECTED c4 c3))
(:goal (and)))
; <magic_json> {"rddl": true}

