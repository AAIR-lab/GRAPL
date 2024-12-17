(define (domain cafeWorld)
    (:requirements :typing :strips :adl :equality)
    (:types
        can
        manipulator
        robot
        location
    )
    (:predicates
        (empty ?gripper - manipulator)
        (ingripper ?obj - can ?gripper - manipulator)
        (at ?loc - location ?r - robot)
        (teleported ?loc - location ?r - robot)
        (order ?obj - can ?loc - location)
        
        (teleport ?loc - location ?r - robot)
        (move ?from - location ?to - location ?r - robot)
        (grasp ?g - manipulator ?loc - location ?obj - can ?r - robot)
        (put ?g - manipulator ?loc - location ?obj - can ?r - robot)
    )

    ; (:actions teleport move grasp put)

    (:action teleport
        :parameters (?loc - location ?r - robot)
        :precondition (and (not (teleported ?loc ?r)) (teleport ?loc ?r))
        :effect (and
            (at ?loc ?r)
            (teleported ?loc ?r)
        )
    )

    (:action move
        :parameters (?from - location ?to - location ?r - robot)
        :precondition (and (at ?from ?r) (move ?from ?to ?r))
        :effect (and
            (not (at ?from ?r))
            (at ?to ?r)
        )
    )
    
    (:action grasp
        :parameters (?g - manipulator ?loc - location ?obj - can ?r - robot)
        :precondition (and
            (empty ?g)
            (at ?loc ?r)
    		(order ?obj ?loc)
    		(grasp ?g ?loc ?obj ?r)
        )
        :effect (and (probabilistic
                    0.8 (and (not (empty ?g)) (ingripper ?obj ?g) (not (order ?obj ?loc)))
                    0.2 (and (empty ?g))
                )
        )
    )
    
    (:action put
        :parameters (?g - manipulator ?loc - location ?obj - can ?r - robot)
        :precondition(and
            (ingripper ?obj ?g)
            (at ?loc ?r)
            (put ?g ?loc ?obj ?r)
        )
        :effect(and
            (not (ingripper ?obj ?g))
            (empty ?g)
            (order ?obj ?loc)
        )
    )
)
