(define (problem problem_0)
(:domain crossing_traffic_mdp)
(:objects
    y1 - ypos
    y2 - ypos
    y3 - ypos
    y4 - ypos
    x1 - xpos
    x2 - xpos
    x3 - xpos)
(:init
    (robot-at x3 y1)
    (obstacle-at x2 y2)
    (obstacle-at x2 y3)
    (NORTH y1 y2)
    (SOUTH y2 y1)
    (NORTH y2 y3)
    (SOUTH y3 y2)
    (NORTH y3 y4)
    (SOUTH y4 y3)
    (EAST x1 x2)
    (WEST x2 x1)
    (EAST x2 x3)
    (WEST x3 x2)
    (MIN-XPOS x1)
    (MAX-XPOS x3)
    (MIN-YPOS y1)
    (MAX-YPOS y4)
    (GOAL x3 y4))
(:goal (and)))
; <magic_json> {"rddl": true}

