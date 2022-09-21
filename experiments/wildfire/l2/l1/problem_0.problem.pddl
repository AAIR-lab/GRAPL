(define (problem problem_0)
(:domain wildfire_mdp)
(:objects
    x1 - x_pos
    x2 - x_pos
    x3 - x_pos
    y1 - y_pos
    y2 - y_pos
    y3 - y_pos)
(:init
    (burning x1 y1)
    (NEIGHBOR x1 y1 x1 y2)
    (NEIGHBOR x1 y1 x2 y1)
    (NEIGHBOR x1 y1 x2 y2)
    (NEIGHBOR x1 y2 x1 y3)
    (NEIGHBOR x1 y2 x2 y1)
    (NEIGHBOR x1 y2 x2 y2)
    (NEIGHBOR x1 y2 x2 y3)
    (NEIGHBOR x1 y3 x2 y2)
    (NEIGHBOR x1 y3 x2 y3)
    (NEIGHBOR x2 y1 x1 y1)
    (NEIGHBOR x2 y1 x1 y2)
    (NEIGHBOR x2 y1 x2 y2)
    (NEIGHBOR x2 y1 x3 y1)
    (NEIGHBOR x2 y2 x1 y1)
    (NEIGHBOR x2 y2 x1 y2)
    (NEIGHBOR x2 y2 x2 y1)
    (NEIGHBOR x2 y2 x2 y3)
    (NEIGHBOR x2 y2 x3 y1)
    (NEIGHBOR x2 y2 x3 y2)
    (NEIGHBOR x2 y2 x3 y3)
    (NEIGHBOR x2 y3 x1 y2)
    (NEIGHBOR x2 y3 x1 y3)
    (NEIGHBOR x2 y3 x2 y2)
    (NEIGHBOR x2 y3 x3 y2)
    (NEIGHBOR x2 y3 x3 y3)
    (NEIGHBOR x3 y1 x2 y1)
    (NEIGHBOR x3 y1 x3 y2)
    (NEIGHBOR x3 y2 x2 y1)
    (NEIGHBOR x3 y2 x2 y2)
    (NEIGHBOR x3 y2 x2 y3)
    (NEIGHBOR x3 y2 x3 y1)
    (NEIGHBOR x3 y2 x3 y3)
    (NEIGHBOR x3 y3 x2 y2)
    (NEIGHBOR x3 y3 x2 y3))
(:goal (and)))
; <magic_json> {"rddl": true}

