(define (problem problem_0)
(:domain crossing_traffic_mdp)
(:objects
    y1 - ypos
    y2 - ypos
    y3 - ypos
    x1 - xpos
    x2 - xpos
    x3 - xpos)
(:init
    (robot-at x3 y1)
    (obstacle-at x1 y2)
    (NORTH y1 y2)
    (SOUTH y2 y1)
    (NORTH y2 y3)
    (SOUTH y3 y2)
    (EAST x1 x2)
    (WEST x2 x1)
    (EAST x2 x3)
    (WEST x3 x2)
    (MIN-XPOS x1)
    (MAX-XPOS x3)
    (MIN-YPOS y1)
    (MAX-YPOS y3)
    (GOAL x3 y3))
(:goal (and)))
; <magic_json> {"rddl": true}

