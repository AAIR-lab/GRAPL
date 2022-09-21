(define (problem problem_0)
(:domain crossing_traffic_mdp)
(:objects
    y1 - ypos
    y2 - ypos
    y3 - ypos
    y4 - ypos
    y5 - ypos
    y6 - ypos
    x1 - xpos
    x2 - xpos
    x3 - xpos
    x4 - xpos)
(:init
    (robot-at x4 y1)
    (obstacle-at x1 y2)
    (obstacle-at x1 y3)
    (obstacle-at x1 y4)
    (obstacle-at x2 y2)
    (obstacle-at x3 y3)
    (obstacle-at x3 y4)
    (NORTH y1 y2)
    (SOUTH y2 y1)
    (NORTH y2 y3)
    (SOUTH y3 y2)
    (NORTH y3 y4)
    (SOUTH y4 y3)
    (NORTH y4 y5)
    (SOUTH y5 y4)
    (NORTH y5 y6)
    (SOUTH y6 y5)
    (EAST x1 x2)
    (WEST x2 x1)
    (EAST x2 x3)
    (WEST x3 x2)
    (EAST x3 x4)
    (WEST x4 x3)
    (MIN-XPOS x1)
    (MAX-XPOS x4)
    (MIN-YPOS y1)
    (MAX-YPOS y6)
    (GOAL x4 y6))
(:goal (and)))
; <magic_json> {"rddl": true}

