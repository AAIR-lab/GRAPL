(define (problem problem_0)
(:domain game_of_life_2_mdp)
(:objects
    l11 - location
    l12 - location
    l13 - location
    l21 - location
    l22 - location
    l23 - location
    l31 - location
    l32 - location
    l33 - location)
(:init
    (alive l11)
    (alive l12)
    (alive l13)
    (alive l21)
    (alive l23)
    (alive l31)
    (NEIGHBOR l11 l12)
    (NEIGHBOR l11 l21)
    (NEIGHBOR l11 l22)
    (NEIGHBOR l12 l11)
    (NEIGHBOR l12 l13)
    (NEIGHBOR l12 l21)
    (NEIGHBOR l12 l22)
    (NEIGHBOR l12 l23)
    (NEIGHBOR l13 l12)
    (NEIGHBOR l13 l22)
    (NEIGHBOR l13 l23)
    (NEIGHBOR l21 l11)
    (NEIGHBOR l21 l12)
    (NEIGHBOR l21 l22)
    (NEIGHBOR l21 l31)
    (NEIGHBOR l21 l32)
    (NEIGHBOR l22 l11)
    (NEIGHBOR l22 l12)
    (NEIGHBOR l22 l13)
    (NEIGHBOR l22 l21)
    (NEIGHBOR l22 l23)
    (NEIGHBOR l22 l31)
    (NEIGHBOR l22 l32)
    (NEIGHBOR l22 l33)
    (NEIGHBOR l23 l12)
    (NEIGHBOR l23 l13)
    (NEIGHBOR l23 l22)
    (NEIGHBOR l23 l32)
    (NEIGHBOR l23 l33)
    (NEIGHBOR l31 l21)
    (NEIGHBOR l31 l22)
    (NEIGHBOR l31 l32)
    (NEIGHBOR l32 l21)
    (NEIGHBOR l32 l22)
    (NEIGHBOR l32 l23)
    (NEIGHBOR l32 l31)
    (NEIGHBOR l32 l33)
    (NEIGHBOR l33 l22)
    (NEIGHBOR l33 l23)
    (NEIGHBOR l33 l32))
(:goal (and)))
; <magic_json> {"rddl": true}

