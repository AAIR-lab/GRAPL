(define (problem explodingblocks)
    (:domain explodingblocks)
    (:objects 
        d - block
        b - block
        a - block
        c - block
        e - block
        f - block
        robot - robot
    )
    (:init
        (clear a)
        (on a b)
        (on b c)
        (on c d)
        (on d e)
        (on e f)
        (ontable f)
        (handempty robot)
    )
    (:goal (and (on f e) (on e d) (on c b) (on b a)))
)
