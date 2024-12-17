(define (problem explodingblocks)
    (:domain explodingblocks)
    (:objects 
        d - block
        b - block
        a - block
        c - block
        e - block
        robot - robot
    )
    (:init
        (clear a)
        (on a b)
        (on b c)
        (on c d)
        (on d e)
        (ontable e)
        (handempty robot)

    )
    (:goal (and (on e d) (on d c) (on c b) (on b a)))
)
