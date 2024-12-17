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
        (ontable c)
        (ontable e)
        (clear e)
        (holding d)
        (handfull robot)



    )
    (:goal (and (on b a) (on c d) (on a c)))
)
