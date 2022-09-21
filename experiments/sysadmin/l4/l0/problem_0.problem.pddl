(define (problem problem_0)
(:domain sysadmin_mdp)
(:objects
    c1 - computer
    c2 - computer
    c3 - computer)
(:init
    (running c1)
    (running c2)
    (running c3)
    (CONNECTED c1 c2)
    (CONNECTED c2 c3)
    (CONNECTED c3 c1))
(:goal (and)))
; <magic_json> {"rddl": true}

