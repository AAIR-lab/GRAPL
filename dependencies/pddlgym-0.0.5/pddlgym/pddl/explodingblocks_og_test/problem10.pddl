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
        (ontable a)
        (clear b)
        (ontable b)
        (clear c)
        (ontable c)
        (clear d)
        (ontable d)
        (clear e)
        (ontable e)
        (clear f)
        (ontable f)
        (handempty robot)
    )
    (:goal (and (on a b) (on b c) (on d e) (on e f)))
)
