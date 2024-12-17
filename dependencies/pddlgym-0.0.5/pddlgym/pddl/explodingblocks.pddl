; Probabilistic version of blocksworld, with explosions

(define (domain explodingblocks)
    (:requirements :strips :typing)
    (:types block robot)
    (:predicates 
        (on ?x - block ?y - block)
        (ontable ?x - block)
        (clear ?x - block)
        (handempty ?x - robot)
        (handfull ?x - robot)
        (holding ?x - block)
        (pick-up ?x - block ?robot - robot)
        (put-down ?x - block ?robot - robot)
        (stack ?x - block ?y - block  ?robot - robot)
        (unstack ?x - block ?y - block ?robot - robot)
        (destroyed ?x - block)
        (table-destroyed)
    )

    ; (:actions pick-up put-down stack unstack)

    (:action pick-up
        :parameters (?x - block ?robot - robot)
        :precondition (and
            (pick-up ?x ?robot) 
            (clear ?x) 
            (ontable ?x) 
            (handempty ?robot)
            (not (destroyed ?x))
            (not (table-destroyed))
        )
        :effect (and
            (not (ontable ?x))
            (not (clear ?x))
            (not (handempty ?robot))
            (handfull ?robot)
            (holding ?x)
        )
    )

    (:action put-down
        :parameters (?x - block ?robot - robot)
        :precondition (and 
            (put-down ?x ?robot)
            (holding ?x)
            (handfull ?robot)
            (not (table-destroyed))
        )
        :effect (and 
            (not (holding ?x))
            (clear ?x)
            (handempty ?robot)
            (not (handfull ?robot))
            (ontable ?x)
            (probabilistic 0.1 (and (table-destroyed)))
        )
    )

    (:action stack
        :parameters (?x - block ?y - block ?robot - robot)
        :precondition (and
            (stack ?x ?y ?robot)
            (holding ?x) 
            (clear ?y)
            (handfull ?robot)
            (not (destroyed ?y))
            (not (table-destroyed))
        )
        :effect (and 
            (not (holding ?x))
            (not (clear ?y))
            (clear ?x)
            (handempty ?robot)
            (not (handfull ?robot))
            (on ?x ?y)
            (probabilistic 0.1 (and (destroyed ?y)))
        )
    )

    (:action unstack
        :parameters (?x - block ?y - block ?robot - robot)
        :precondition (and
            (unstack ?x ?y ?robot)
            (on ?x ?y)
            (clear ?x)
            (handempty ?robot)
            (not (destroyed ?x))
            (not (table-destroyed))
        )
        :effect (and 
            (holding ?x)
            (clear ?y)
            (not (clear ?x))
            (not (handempty ?robot))
            (handfull ?robot)
            (not (on ?x ?y))
        )
    )
)
