(define (problem problem_0)
(:domain game_of_life_mdp)
(:objects
    x1 - x_pos
    x2 - x_pos
    x3 - x_pos
    x4 - x_pos
    x5 - x_pos
    x6 - x_pos
    x7 - x_pos
    x8 - x_pos
    x9 - x_pos
    y1 - y_pos
    y2 - y_pos
    y3 - y_pos
    y4 - y_pos
    y5 - y_pos
    y6 - y_pos
    y7 - y_pos
    y8 - y_pos
    y9 - y_pos)
(:init
    (alive x1 y1)
    (alive x1 y9)
    (alive x2 y4)
    (alive x2 y6)
    (alive x3 y1)
    (alive x3 y3)
    (alive x3 y4)
    (alive x3 y6)
    (alive x3 y7)
    (alive x4 y1)
    (alive x4 y2)
    (alive x4 y6)
    (alive x4 y7)
    (alive x4 y8)
    (alive x4 y9)
    (alive x5 y1)
    (alive x5 y2)
    (alive x5 y4)
    (alive x5 y6)
    (alive x5 y7)
    (alive x5 y9)
    (alive x6 y2)
    (alive x6 y3)
    (alive x6 y4)
    (alive x6 y5)
    (alive x6 y7)
    (alive x6 y8)
    (alive x6 y9)
    (alive x7 y1)
    (alive x7 y4)
    (alive x7 y6)
    (alive x7 y7)
    (alive x7 y9)
    (alive x8 y3)
    (alive x8 y4)
    (alive x8 y5)
    (alive x8 y7)
    (alive x8 y8)
    (alive x9 y1)
    (alive x9 y2)
    (alive x9 y4)
    (NEIGHBOR x1 y1 x1 y2)
    (NEIGHBOR x1 y1 x2 y1)
    (NEIGHBOR x1 y1 x2 y2)
    (NEIGHBOR x1 y2 x1 y1)
    (NEIGHBOR x1 y2 x1 y3)
    (NEIGHBOR x1 y2 x2 y1)
    (NEIGHBOR x1 y2 x2 y2)
    (NEIGHBOR x1 y2 x2 y3)
    (NEIGHBOR x1 y3 x1 y2)
    (NEIGHBOR x1 y3 x1 y4)
    (NEIGHBOR x1 y3 x2 y2)
    (NEIGHBOR x1 y3 x2 y3)
    (NEIGHBOR x1 y3 x2 y4)
    (NEIGHBOR x1 y4 x1 y3)
    (NEIGHBOR x1 y4 x1 y5)
    (NEIGHBOR x1 y4 x2 y3)
    (NEIGHBOR x1 y4 x2 y4)
    (NEIGHBOR x1 y4 x2 y5)
    (NEIGHBOR x1 y5 x1 y4)
    (NEIGHBOR x1 y5 x1 y6)
    (NEIGHBOR x1 y5 x2 y4)
    (NEIGHBOR x1 y5 x2 y5)
    (NEIGHBOR x1 y5 x2 y6)
    (NEIGHBOR x1 y6 x1 y5)
    (NEIGHBOR x1 y6 x1 y7)
    (NEIGHBOR x1 y6 x2 y5)
    (NEIGHBOR x1 y6 x2 y6)
    (NEIGHBOR x1 y6 x2 y7)
    (NEIGHBOR x1 y7 x1 y6)
    (NEIGHBOR x1 y7 x1 y8)
    (NEIGHBOR x1 y7 x2 y6)
    (NEIGHBOR x1 y7 x2 y7)
    (NEIGHBOR x1 y7 x2 y8)
    (NEIGHBOR x1 y8 x1 y7)
    (NEIGHBOR x1 y8 x1 y9)
    (NEIGHBOR x1 y8 x2 y7)
    (NEIGHBOR x1 y8 x2 y8)
    (NEIGHBOR x1 y8 x2 y9)
    (NEIGHBOR x1 y9 x1 y8)
    (NEIGHBOR x1 y9 x2 y8)
    (NEIGHBOR x1 y9 x2 y9)
    (NEIGHBOR x2 y1 x1 y1)
    (NEIGHBOR x2 y1 x1 y2)
    (NEIGHBOR x2 y1 x2 y2)
    (NEIGHBOR x2 y1 x3 y1)
    (NEIGHBOR x2 y1 x3 y2)
    (NEIGHBOR x2 y2 x1 y1)
    (NEIGHBOR x2 y2 x1 y2)
    (NEIGHBOR x2 y2 x1 y3)
    (NEIGHBOR x2 y2 x2 y1)
    (NEIGHBOR x2 y2 x2 y3)
    (NEIGHBOR x2 y2 x3 y1)
    (NEIGHBOR x2 y2 x3 y2)
    (NEIGHBOR x2 y2 x3 y3)
    (NEIGHBOR x2 y3 x1 y2)
    (NEIGHBOR x2 y3 x1 y3)
    (NEIGHBOR x2 y3 x1 y4)
    (NEIGHBOR x2 y3 x2 y2)
    (NEIGHBOR x2 y3 x2 y4)
    (NEIGHBOR x2 y3 x3 y2)
    (NEIGHBOR x2 y3 x3 y3)
    (NEIGHBOR x2 y3 x3 y4)
    (NEIGHBOR x2 y4 x1 y3)
    (NEIGHBOR x2 y4 x1 y4)
    (NEIGHBOR x2 y4 x1 y5)
    (NEIGHBOR x2 y4 x2 y3)
    (NEIGHBOR x2 y4 x2 y5)
    (NEIGHBOR x2 y4 x3 y3)
    (NEIGHBOR x2 y4 x3 y4)
    (NEIGHBOR x2 y4 x3 y5)
    (NEIGHBOR x2 y5 x1 y4)
    (NEIGHBOR x2 y5 x1 y5)
    (NEIGHBOR x2 y5 x1 y6)
    (NEIGHBOR x2 y5 x2 y4)
    (NEIGHBOR x2 y5 x2 y6)
    (NEIGHBOR x2 y5 x3 y4)
    (NEIGHBOR x2 y5 x3 y5)
    (NEIGHBOR x2 y5 x3 y6)
    (NEIGHBOR x2 y6 x1 y5)
    (NEIGHBOR x2 y6 x1 y6)
    (NEIGHBOR x2 y6 x1 y7)
    (NEIGHBOR x2 y6 x2 y5)
    (NEIGHBOR x2 y6 x2 y7)
    (NEIGHBOR x2 y6 x3 y5)
    (NEIGHBOR x2 y6 x3 y6)
    (NEIGHBOR x2 y6 x3 y7)
    (NEIGHBOR x2 y7 x1 y6)
    (NEIGHBOR x2 y7 x1 y7)
    (NEIGHBOR x2 y7 x1 y8)
    (NEIGHBOR x2 y7 x2 y6)
    (NEIGHBOR x2 y7 x2 y8)
    (NEIGHBOR x2 y7 x3 y6)
    (NEIGHBOR x2 y7 x3 y7)
    (NEIGHBOR x2 y7 x3 y8)
    (NEIGHBOR x2 y8 x1 y7)
    (NEIGHBOR x2 y8 x1 y8)
    (NEIGHBOR x2 y8 x1 y9)
    (NEIGHBOR x2 y8 x2 y7)
    (NEIGHBOR x2 y8 x2 y9)
    (NEIGHBOR x2 y8 x3 y7)
    (NEIGHBOR x2 y8 x3 y8)
    (NEIGHBOR x2 y8 x3 y9)
    (NEIGHBOR x2 y9 x1 y8)
    (NEIGHBOR x2 y9 x1 y9)
    (NEIGHBOR x2 y9 x2 y8)
    (NEIGHBOR x2 y9 x3 y8)
    (NEIGHBOR x2 y9 x3 y9)
    (NEIGHBOR x3 y1 x2 y1)
    (NEIGHBOR x3 y1 x2 y2)
    (NEIGHBOR x3 y1 x3 y2)
    (NEIGHBOR x3 y1 x4 y1)
    (NEIGHBOR x3 y1 x4 y2)
    (NEIGHBOR x3 y2 x2 y1)
    (NEIGHBOR x3 y2 x2 y2)
    (NEIGHBOR x3 y2 x2 y3)
    (NEIGHBOR x3 y2 x3 y1)
    (NEIGHBOR x3 y2 x3 y3)
    (NEIGHBOR x3 y2 x4 y1)
    (NEIGHBOR x3 y2 x4 y2)
    (NEIGHBOR x3 y2 x4 y3)
    (NEIGHBOR x3 y3 x2 y2)
    (NEIGHBOR x3 y3 x2 y3)
    (NEIGHBOR x3 y3 x2 y4)
    (NEIGHBOR x3 y3 x3 y2)
    (NEIGHBOR x3 y3 x3 y4)
    (NEIGHBOR x3 y3 x4 y2)
    (NEIGHBOR x3 y3 x4 y3)
    (NEIGHBOR x3 y3 x4 y4)
    (NEIGHBOR x3 y4 x2 y3)
    (NEIGHBOR x3 y4 x2 y4)
    (NEIGHBOR x3 y4 x2 y5)
    (NEIGHBOR x3 y4 x3 y3)
    (NEIGHBOR x3 y4 x3 y5)
    (NEIGHBOR x3 y4 x4 y3)
    (NEIGHBOR x3 y4 x4 y4)
    (NEIGHBOR x3 y4 x4 y5)
    (NEIGHBOR x3 y5 x2 y4)
    (NEIGHBOR x3 y5 x2 y5)
    (NEIGHBOR x3 y5 x2 y6)
    (NEIGHBOR x3 y5 x3 y4)
    (NEIGHBOR x3 y5 x3 y6)
    (NEIGHBOR x3 y5 x4 y4)
    (NEIGHBOR x3 y5 x4 y5)
    (NEIGHBOR x3 y5 x4 y6)
    (NEIGHBOR x3 y6 x2 y5)
    (NEIGHBOR x3 y6 x2 y6)
    (NEIGHBOR x3 y6 x2 y7)
    (NEIGHBOR x3 y6 x3 y5)
    (NEIGHBOR x3 y6 x3 y7)
    (NEIGHBOR x3 y6 x4 y5)
    (NEIGHBOR x3 y6 x4 y6)
    (NEIGHBOR x3 y6 x4 y7)
    (NEIGHBOR x3 y7 x2 y6)
    (NEIGHBOR x3 y7 x2 y7)
    (NEIGHBOR x3 y7 x2 y8)
    (NEIGHBOR x3 y7 x3 y6)
    (NEIGHBOR x3 y7 x3 y8)
    (NEIGHBOR x3 y7 x4 y6)
    (NEIGHBOR x3 y7 x4 y7)
    (NEIGHBOR x3 y7 x4 y8)
    (NEIGHBOR x3 y8 x2 y7)
    (NEIGHBOR x3 y8 x2 y8)
    (NEIGHBOR x3 y8 x2 y9)
    (NEIGHBOR x3 y8 x3 y7)
    (NEIGHBOR x3 y8 x3 y9)
    (NEIGHBOR x3 y8 x4 y7)
    (NEIGHBOR x3 y8 x4 y8)
    (NEIGHBOR x3 y8 x4 y9)
    (NEIGHBOR x3 y9 x2 y8)
    (NEIGHBOR x3 y9 x2 y9)
    (NEIGHBOR x3 y9 x3 y8)
    (NEIGHBOR x3 y9 x4 y8)
    (NEIGHBOR x3 y9 x4 y9)
    (NEIGHBOR x4 y1 x3 y1)
    (NEIGHBOR x4 y1 x3 y2)
    (NEIGHBOR x4 y1 x4 y2)
    (NEIGHBOR x4 y1 x5 y1)
    (NEIGHBOR x4 y1 x5 y2)
    (NEIGHBOR x4 y2 x3 y1)
    (NEIGHBOR x4 y2 x3 y2)
    (NEIGHBOR x4 y2 x3 y3)
    (NEIGHBOR x4 y2 x4 y1)
    (NEIGHBOR x4 y2 x4 y3)
    (NEIGHBOR x4 y2 x5 y1)
    (NEIGHBOR x4 y2 x5 y2)
    (NEIGHBOR x4 y2 x5 y3)
    (NEIGHBOR x4 y3 x3 y2)
    (NEIGHBOR x4 y3 x3 y3)
    (NEIGHBOR x4 y3 x3 y4)
    (NEIGHBOR x4 y3 x4 y2)
    (NEIGHBOR x4 y3 x4 y4)
    (NEIGHBOR x4 y3 x5 y2)
    (NEIGHBOR x4 y3 x5 y3)
    (NEIGHBOR x4 y3 x5 y4)
    (NEIGHBOR x4 y4 x3 y3)
    (NEIGHBOR x4 y4 x3 y4)
    (NEIGHBOR x4 y4 x3 y5)
    (NEIGHBOR x4 y4 x4 y3)
    (NEIGHBOR x4 y4 x4 y5)
    (NEIGHBOR x4 y4 x5 y3)
    (NEIGHBOR x4 y4 x5 y4)
    (NEIGHBOR x4 y4 x5 y5)
    (NEIGHBOR x4 y5 x3 y4)
    (NEIGHBOR x4 y5 x3 y5)
    (NEIGHBOR x4 y5 x3 y6)
    (NEIGHBOR x4 y5 x4 y4)
    (NEIGHBOR x4 y5 x4 y6)
    (NEIGHBOR x4 y5 x5 y4)
    (NEIGHBOR x4 y5 x5 y5)
    (NEIGHBOR x4 y5 x5 y6)
    (NEIGHBOR x4 y6 x3 y5)
    (NEIGHBOR x4 y6 x3 y6)
    (NEIGHBOR x4 y6 x3 y7)
    (NEIGHBOR x4 y6 x4 y5)
    (NEIGHBOR x4 y6 x4 y7)
    (NEIGHBOR x4 y6 x5 y5)
    (NEIGHBOR x4 y6 x5 y6)
    (NEIGHBOR x4 y6 x5 y7)
    (NEIGHBOR x4 y7 x3 y6)
    (NEIGHBOR x4 y7 x3 y7)
    (NEIGHBOR x4 y7 x3 y8)
    (NEIGHBOR x4 y7 x4 y6)
    (NEIGHBOR x4 y7 x4 y8)
    (NEIGHBOR x4 y7 x5 y6)
    (NEIGHBOR x4 y7 x5 y7)
    (NEIGHBOR x4 y7 x5 y8)
    (NEIGHBOR x4 y8 x3 y7)
    (NEIGHBOR x4 y8 x3 y8)
    (NEIGHBOR x4 y8 x3 y9)
    (NEIGHBOR x4 y8 x4 y7)
    (NEIGHBOR x4 y8 x4 y9)
    (NEIGHBOR x4 y8 x5 y7)
    (NEIGHBOR x4 y8 x5 y8)
    (NEIGHBOR x4 y8 x5 y9)
    (NEIGHBOR x4 y9 x3 y8)
    (NEIGHBOR x4 y9 x3 y9)
    (NEIGHBOR x4 y9 x4 y8)
    (NEIGHBOR x4 y9 x5 y8)
    (NEIGHBOR x4 y9 x5 y9)
    (NEIGHBOR x5 y1 x4 y1)
    (NEIGHBOR x5 y1 x4 y2)
    (NEIGHBOR x5 y1 x5 y2)
    (NEIGHBOR x5 y1 x6 y1)
    (NEIGHBOR x5 y1 x6 y2)
    (NEIGHBOR x5 y2 x4 y1)
    (NEIGHBOR x5 y2 x4 y2)
    (NEIGHBOR x5 y2 x4 y3)
    (NEIGHBOR x5 y2 x5 y1)
    (NEIGHBOR x5 y2 x5 y3)
    (NEIGHBOR x5 y2 x6 y1)
    (NEIGHBOR x5 y2 x6 y2)
    (NEIGHBOR x5 y2 x6 y3)
    (NEIGHBOR x5 y3 x4 y2)
    (NEIGHBOR x5 y3 x4 y3)
    (NEIGHBOR x5 y3 x4 y4)
    (NEIGHBOR x5 y3 x5 y2)
    (NEIGHBOR x5 y3 x5 y4)
    (NEIGHBOR x5 y3 x6 y2)
    (NEIGHBOR x5 y3 x6 y3)
    (NEIGHBOR x5 y3 x6 y4)
    (NEIGHBOR x5 y4 x4 y3)
    (NEIGHBOR x5 y4 x4 y4)
    (NEIGHBOR x5 y4 x4 y5)
    (NEIGHBOR x5 y4 x5 y3)
    (NEIGHBOR x5 y4 x5 y5)
    (NEIGHBOR x5 y4 x6 y3)
    (NEIGHBOR x5 y4 x6 y4)
    (NEIGHBOR x5 y4 x6 y5)
    (NEIGHBOR x5 y5 x4 y4)
    (NEIGHBOR x5 y5 x4 y5)
    (NEIGHBOR x5 y5 x4 y6)
    (NEIGHBOR x5 y5 x5 y4)
    (NEIGHBOR x5 y5 x5 y6)
    (NEIGHBOR x5 y5 x6 y4)
    (NEIGHBOR x5 y5 x6 y5)
    (NEIGHBOR x5 y5 x6 y6)
    (NEIGHBOR x5 y6 x4 y5)
    (NEIGHBOR x5 y6 x4 y6)
    (NEIGHBOR x5 y6 x4 y7)
    (NEIGHBOR x5 y6 x5 y5)
    (NEIGHBOR x5 y6 x5 y7)
    (NEIGHBOR x5 y6 x6 y5)
    (NEIGHBOR x5 y6 x6 y6)
    (NEIGHBOR x5 y6 x6 y7)
    (NEIGHBOR x5 y7 x4 y6)
    (NEIGHBOR x5 y7 x4 y7)
    (NEIGHBOR x5 y7 x4 y8)
    (NEIGHBOR x5 y7 x5 y6)
    (NEIGHBOR x5 y7 x5 y8)
    (NEIGHBOR x5 y7 x6 y6)
    (NEIGHBOR x5 y7 x6 y7)
    (NEIGHBOR x5 y7 x6 y8)
    (NEIGHBOR x5 y8 x4 y7)
    (NEIGHBOR x5 y8 x4 y8)
    (NEIGHBOR x5 y8 x4 y9)
    (NEIGHBOR x5 y8 x5 y7)
    (NEIGHBOR x5 y8 x5 y9)
    (NEIGHBOR x5 y8 x6 y7)
    (NEIGHBOR x5 y8 x6 y8)
    (NEIGHBOR x5 y8 x6 y9)
    (NEIGHBOR x5 y9 x4 y8)
    (NEIGHBOR x5 y9 x4 y9)
    (NEIGHBOR x5 y9 x5 y8)
    (NEIGHBOR x5 y9 x6 y8)
    (NEIGHBOR x5 y9 x6 y9)
    (NEIGHBOR x6 y1 x5 y1)
    (NEIGHBOR x6 y1 x5 y2)
    (NEIGHBOR x6 y1 x6 y2)
    (NEIGHBOR x6 y1 x7 y1)
    (NEIGHBOR x6 y1 x7 y2)
    (NEIGHBOR x6 y2 x5 y1)
    (NEIGHBOR x6 y2 x5 y2)
    (NEIGHBOR x6 y2 x5 y3)
    (NEIGHBOR x6 y2 x6 y1)
    (NEIGHBOR x6 y2 x6 y3)
    (NEIGHBOR x6 y2 x7 y1)
    (NEIGHBOR x6 y2 x7 y2)
    (NEIGHBOR x6 y2 x7 y3)
    (NEIGHBOR x6 y3 x5 y2)
    (NEIGHBOR x6 y3 x5 y3)
    (NEIGHBOR x6 y3 x5 y4)
    (NEIGHBOR x6 y3 x6 y2)
    (NEIGHBOR x6 y3 x6 y4)
    (NEIGHBOR x6 y3 x7 y2)
    (NEIGHBOR x6 y3 x7 y3)
    (NEIGHBOR x6 y3 x7 y4)
    (NEIGHBOR x6 y4 x5 y3)
    (NEIGHBOR x6 y4 x5 y4)
    (NEIGHBOR x6 y4 x5 y5)
    (NEIGHBOR x6 y4 x6 y3)
    (NEIGHBOR x6 y4 x6 y5)
    (NEIGHBOR x6 y4 x7 y3)
    (NEIGHBOR x6 y4 x7 y4)
    (NEIGHBOR x6 y4 x7 y5)
    (NEIGHBOR x6 y5 x5 y4)
    (NEIGHBOR x6 y5 x5 y5)
    (NEIGHBOR x6 y5 x5 y6)
    (NEIGHBOR x6 y5 x6 y4)
    (NEIGHBOR x6 y5 x6 y6)
    (NEIGHBOR x6 y5 x7 y4)
    (NEIGHBOR x6 y5 x7 y5)
    (NEIGHBOR x6 y5 x7 y6)
    (NEIGHBOR x6 y6 x5 y5)
    (NEIGHBOR x6 y6 x5 y6)
    (NEIGHBOR x6 y6 x5 y7)
    (NEIGHBOR x6 y6 x6 y5)
    (NEIGHBOR x6 y6 x6 y7)
    (NEIGHBOR x6 y6 x7 y5)
    (NEIGHBOR x6 y6 x7 y6)
    (NEIGHBOR x6 y6 x7 y7)
    (NEIGHBOR x6 y7 x5 y6)
    (NEIGHBOR x6 y7 x5 y7)
    (NEIGHBOR x6 y7 x5 y8)
    (NEIGHBOR x6 y7 x6 y6)
    (NEIGHBOR x6 y7 x6 y8)
    (NEIGHBOR x6 y7 x7 y6)
    (NEIGHBOR x6 y7 x7 y7)
    (NEIGHBOR x6 y7 x7 y8)
    (NEIGHBOR x6 y8 x5 y7)
    (NEIGHBOR x6 y8 x5 y8)
    (NEIGHBOR x6 y8 x5 y9)
    (NEIGHBOR x6 y8 x6 y7)
    (NEIGHBOR x6 y8 x6 y9)
    (NEIGHBOR x6 y8 x7 y7)
    (NEIGHBOR x6 y8 x7 y8)
    (NEIGHBOR x6 y8 x7 y9)
    (NEIGHBOR x6 y9 x5 y8)
    (NEIGHBOR x6 y9 x5 y9)
    (NEIGHBOR x6 y9 x6 y8)
    (NEIGHBOR x6 y9 x7 y8)
    (NEIGHBOR x6 y9 x7 y9)
    (NEIGHBOR x7 y1 x6 y1)
    (NEIGHBOR x7 y1 x6 y2)
    (NEIGHBOR x7 y1 x7 y2)
    (NEIGHBOR x7 y1 x8 y1)
    (NEIGHBOR x7 y1 x8 y2)
    (NEIGHBOR x7 y2 x6 y1)
    (NEIGHBOR x7 y2 x6 y2)
    (NEIGHBOR x7 y2 x6 y3)
    (NEIGHBOR x7 y2 x7 y1)
    (NEIGHBOR x7 y2 x7 y3)
    (NEIGHBOR x7 y2 x8 y1)
    (NEIGHBOR x7 y2 x8 y2)
    (NEIGHBOR x7 y2 x8 y3)
    (NEIGHBOR x7 y3 x6 y2)
    (NEIGHBOR x7 y3 x6 y3)
    (NEIGHBOR x7 y3 x6 y4)
    (NEIGHBOR x7 y3 x7 y2)
    (NEIGHBOR x7 y3 x7 y4)
    (NEIGHBOR x7 y3 x8 y2)
    (NEIGHBOR x7 y3 x8 y3)
    (NEIGHBOR x7 y3 x8 y4)
    (NEIGHBOR x7 y4 x6 y3)
    (NEIGHBOR x7 y4 x6 y4)
    (NEIGHBOR x7 y4 x6 y5)
    (NEIGHBOR x7 y4 x7 y3)
    (NEIGHBOR x7 y4 x7 y5)
    (NEIGHBOR x7 y4 x8 y3)
    (NEIGHBOR x7 y4 x8 y4)
    (NEIGHBOR x7 y4 x8 y5)
    (NEIGHBOR x7 y5 x6 y4)
    (NEIGHBOR x7 y5 x6 y5)
    (NEIGHBOR x7 y5 x6 y6)
    (NEIGHBOR x7 y5 x7 y4)
    (NEIGHBOR x7 y5 x7 y6)
    (NEIGHBOR x7 y5 x8 y4)
    (NEIGHBOR x7 y5 x8 y5)
    (NEIGHBOR x7 y5 x8 y6)
    (NEIGHBOR x7 y6 x6 y5)
    (NEIGHBOR x7 y6 x6 y6)
    (NEIGHBOR x7 y6 x6 y7)
    (NEIGHBOR x7 y6 x7 y5)
    (NEIGHBOR x7 y6 x7 y7)
    (NEIGHBOR x7 y6 x8 y5)
    (NEIGHBOR x7 y6 x8 y6)
    (NEIGHBOR x7 y6 x8 y7)
    (NEIGHBOR x7 y7 x6 y6)
    (NEIGHBOR x7 y7 x6 y7)
    (NEIGHBOR x7 y7 x6 y8)
    (NEIGHBOR x7 y7 x7 y6)
    (NEIGHBOR x7 y7 x7 y8)
    (NEIGHBOR x7 y7 x8 y6)
    (NEIGHBOR x7 y7 x8 y7)
    (NEIGHBOR x7 y7 x8 y8)
    (NEIGHBOR x7 y8 x6 y7)
    (NEIGHBOR x7 y8 x6 y8)
    (NEIGHBOR x7 y8 x6 y9)
    (NEIGHBOR x7 y8 x7 y7)
    (NEIGHBOR x7 y8 x7 y9)
    (NEIGHBOR x7 y8 x8 y7)
    (NEIGHBOR x7 y8 x8 y8)
    (NEIGHBOR x7 y8 x8 y9)
    (NEIGHBOR x7 y9 x6 y8)
    (NEIGHBOR x7 y9 x6 y9)
    (NEIGHBOR x7 y9 x7 y8)
    (NEIGHBOR x7 y9 x8 y8)
    (NEIGHBOR x7 y9 x8 y9)
    (NEIGHBOR x8 y1 x7 y1)
    (NEIGHBOR x8 y1 x7 y2)
    (NEIGHBOR x8 y1 x8 y2)
    (NEIGHBOR x8 y1 x9 y1)
    (NEIGHBOR x8 y1 x9 y2)
    (NEIGHBOR x8 y2 x7 y1)
    (NEIGHBOR x8 y2 x7 y2)
    (NEIGHBOR x8 y2 x7 y3)
    (NEIGHBOR x8 y2 x8 y1)
    (NEIGHBOR x8 y2 x8 y3)
    (NEIGHBOR x8 y2 x9 y1)
    (NEIGHBOR x8 y2 x9 y2)
    (NEIGHBOR x8 y2 x9 y3)
    (NEIGHBOR x8 y3 x7 y2)
    (NEIGHBOR x8 y3 x7 y3)
    (NEIGHBOR x8 y3 x7 y4)
    (NEIGHBOR x8 y3 x8 y2)
    (NEIGHBOR x8 y3 x8 y4)
    (NEIGHBOR x8 y3 x9 y2)
    (NEIGHBOR x8 y3 x9 y3)
    (NEIGHBOR x8 y3 x9 y4)
    (NEIGHBOR x8 y4 x7 y3)
    (NEIGHBOR x8 y4 x7 y4)
    (NEIGHBOR x8 y4 x7 y5)
    (NEIGHBOR x8 y4 x8 y3)
    (NEIGHBOR x8 y4 x8 y5)
    (NEIGHBOR x8 y4 x9 y3)
    (NEIGHBOR x8 y4 x9 y4)
    (NEIGHBOR x8 y4 x9 y5)
    (NEIGHBOR x8 y5 x7 y4)
    (NEIGHBOR x8 y5 x7 y5)
    (NEIGHBOR x8 y5 x7 y6)
    (NEIGHBOR x8 y5 x8 y4)
    (NEIGHBOR x8 y5 x8 y6)
    (NEIGHBOR x8 y5 x9 y4)
    (NEIGHBOR x8 y5 x9 y5)
    (NEIGHBOR x8 y5 x9 y6)
    (NEIGHBOR x8 y6 x7 y5)
    (NEIGHBOR x8 y6 x7 y6)
    (NEIGHBOR x8 y6 x7 y7)
    (NEIGHBOR x8 y6 x8 y5)
    (NEIGHBOR x8 y6 x8 y7)
    (NEIGHBOR x8 y6 x9 y5)
    (NEIGHBOR x8 y6 x9 y6)
    (NEIGHBOR x8 y6 x9 y7)
    (NEIGHBOR x8 y7 x7 y6)
    (NEIGHBOR x8 y7 x7 y7)
    (NEIGHBOR x8 y7 x7 y8)
    (NEIGHBOR x8 y7 x8 y6)
    (NEIGHBOR x8 y7 x8 y8)
    (NEIGHBOR x8 y7 x9 y6)
    (NEIGHBOR x8 y7 x9 y7)
    (NEIGHBOR x8 y7 x9 y8)
    (NEIGHBOR x8 y8 x7 y7)
    (NEIGHBOR x8 y8 x7 y8)
    (NEIGHBOR x8 y8 x7 y9)
    (NEIGHBOR x8 y8 x8 y7)
    (NEIGHBOR x8 y8 x8 y9)
    (NEIGHBOR x8 y8 x9 y7)
    (NEIGHBOR x8 y8 x9 y8)
    (NEIGHBOR x8 y8 x9 y9)
    (NEIGHBOR x8 y9 x7 y8)
    (NEIGHBOR x8 y9 x7 y9)
    (NEIGHBOR x8 y9 x8 y8)
    (NEIGHBOR x8 y9 x9 y8)
    (NEIGHBOR x8 y9 x9 y9)
    (NEIGHBOR x9 y1 x8 y1)
    (NEIGHBOR x9 y1 x8 y2)
    (NEIGHBOR x9 y1 x9 y2)
    (NEIGHBOR x9 y2 x8 y1)
    (NEIGHBOR x9 y2 x8 y2)
    (NEIGHBOR x9 y2 x8 y3)
    (NEIGHBOR x9 y2 x9 y1)
    (NEIGHBOR x9 y2 x9 y3)
    (NEIGHBOR x9 y3 x8 y2)
    (NEIGHBOR x9 y3 x8 y3)
    (NEIGHBOR x9 y3 x8 y4)
    (NEIGHBOR x9 y3 x9 y2)
    (NEIGHBOR x9 y3 x9 y4)
    (NEIGHBOR x9 y4 x8 y3)
    (NEIGHBOR x9 y4 x8 y4)
    (NEIGHBOR x9 y4 x8 y5)
    (NEIGHBOR x9 y4 x9 y3)
    (NEIGHBOR x9 y4 x9 y5)
    (NEIGHBOR x9 y5 x8 y4)
    (NEIGHBOR x9 y5 x8 y5)
    (NEIGHBOR x9 y5 x8 y6)
    (NEIGHBOR x9 y5 x9 y4)
    (NEIGHBOR x9 y5 x9 y6)
    (NEIGHBOR x9 y6 x8 y5)
    (NEIGHBOR x9 y6 x8 y6)
    (NEIGHBOR x9 y6 x8 y7)
    (NEIGHBOR x9 y6 x9 y5)
    (NEIGHBOR x9 y6 x9 y7)
    (NEIGHBOR x9 y7 x8 y6)
    (NEIGHBOR x9 y7 x8 y7)
    (NEIGHBOR x9 y7 x8 y8)
    (NEIGHBOR x9 y7 x9 y6)
    (NEIGHBOR x9 y7 x9 y8)
    (NEIGHBOR x9 y8 x8 y7)
    (NEIGHBOR x9 y8 x8 y8)
    (NEIGHBOR x9 y8 x8 y9)
    (NEIGHBOR x9 y8 x9 y7)
    (NEIGHBOR x9 y8 x9 y9)
    (NEIGHBOR x9 y9 x8 y8)
    (NEIGHBOR x9 y9 x8 y9)
    (NEIGHBOR x9 y9 x9 y8))
(:goal (and)))
; <magic_json> {"rddl": true}
