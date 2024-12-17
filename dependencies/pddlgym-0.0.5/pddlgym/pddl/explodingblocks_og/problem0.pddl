(define (problem explodingblocks)
    (:domain explodingblocks)
    (:objects 
        b - block
        a - block
        robot - robot
    )
    (:init  
        (clear a) 
        (clear b)   
        (ontable a)
        (ontable b) 
        (handempty robot)


    )
    (:goal (and (on a b)))
)
