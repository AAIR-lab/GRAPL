(define (problem problem_0)
(:domain crossing_traffic_mdp)
(:objects
    y1 - ypos
    y2 - ypos
    y3 - ypos
    y4 - ypos
    y5 - ypos
    y6 - ypos
    y7 - ypos
    x1 - xpos
    x2 - xpos
    x3 - xpos
    x4 - xpos
    x5 - xpos
    x6 - xpos
    x7 - xpos)
(:init
    (robot-at x7 y1)
    (obstacle-at x1 y2)
    (obstacle-at x1 y4)
    (obstacle-at x1 y6)
    (obstacle-at x2 y2)
    (obstacle-at x2 y3)
    (obstacle-at x2 y6)
    (obstacle-at x3 y2)
    (obstacle-at x3 y4)
    (obstacle-at x3 y5)
    (obstacle-at x3 y6)
    (obstacle-at x4 y5)
    (obstacle-at x4 y6)
    (obstacle-at x5 y4)
    (obstacle-at x5 y5)
    (obstacle-at x5 y6)
    (obstacle-at x6 y2)
    (obstacle-at x6 y5)
    (obstacle-at x7 y2)
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
    (NORTH y6 y7)
    (SOUTH y7 y6)
    (EAST x1 x2)
    (WEST x2 x1)
    (EAST x2 x3)
    (WEST x3 x2)
    (EAST x3 x4)
    (WEST x4 x3)
    (EAST x4 x5)
    (WEST x5 x4)
    (EAST x5 x6)
    (WEST x6 x5)
    (EAST x6 x7)
    (WEST x7 x6)
    (MIN-XPOS x1)
    (MAX-XPOS x7)
    (MIN-YPOS y1)
    (MAX-YPOS y7)
    (GOAL x7 y7))
(:goal (and)))
; <magic_json> {"rddl": true}

