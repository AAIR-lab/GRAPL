(define (problem explodingblocks)
    (:domain explodingblocks)
    (:objects 
        d - block
        b - block
        a - block
        c - block
        robot - robot
    )
    (:init 
        (clear b)
        (clear d)
        (on c a)
        (on d c)
        (ontable a)
        (ontable b) 
        (handempty robot)
    )
    (:goal (and (on b a) (on a c) (on c d)))
)
