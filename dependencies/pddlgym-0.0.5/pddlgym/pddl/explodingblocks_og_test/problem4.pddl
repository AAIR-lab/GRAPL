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
        (clear c) 
        (clear a) 
        (clear b) 
        (clear d) 
        (ontable c) 
        (ontable a)
        (ontable b) 
        (ontable e)
        (on d e)
        (handempty robot)
    )
    (:goal (and (on a b) (on b c) (on c d) (on d e)))
)
