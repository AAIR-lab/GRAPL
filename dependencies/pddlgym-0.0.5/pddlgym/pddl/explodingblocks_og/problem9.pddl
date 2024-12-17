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
        (clear b)
        (on b c)
        (on c d)
        (on d e)
        (on e f)
        (ontable f)
        (holding a)
        (handfull robot)


    )
    (:goal (and (on b c) (on c d) (on d e) (on e f) (on f a)))
)
