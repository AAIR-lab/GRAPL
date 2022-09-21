(define (problem problem_0)
(:domain game_of_life_2_mdp)
(:objects
    l11 - location
    l12 - location
    l21 - location
    l22 - location)
(:init
    (alive l11)
    (alive l12)
    (alive l21)
    (alive l22)
    (NEIGHBOR l11 l12)
    (NEIGHBOR l11 l21)
    (NEIGHBOR l11 l22)
    (NEIGHBOR l12 l11)
    (NEIGHBOR l12 l21)
    (NEIGHBOR l12 l22)
    (NEIGHBOR l21 l11)
    (NEIGHBOR l21 l12)
    (NEIGHBOR l21 l22)
    (NEIGHBOR l22 l11)
    (NEIGHBOR l22 l12)
    (NEIGHBOR l22 l21))
(:goal (and)))
; <magic_json> {"rddl": true}

