(define (problem problem_0)
(:domain game_of_life_2_mdp)
(:objects
    l11 - location
    l12 - location
    l13 - location
    l14 - location
    l15 - location
    l21 - location
    l22 - location
    l23 - location
    l24 - location
    l25 - location
    l31 - location
    l32 - location
    l33 - location
    l34 - location
    l35 - location
    l41 - location
    l42 - location
    l43 - location
    l44 - location
    l45 - location
    l51 - location
    l52 - location
    l53 - location
    l54 - location
    l55 - location)
(:init
    (alive l11)
    (alive l12)
    (alive l15)
    (alive l21)
    (alive l23)
    (alive l24)
    (alive l25)
    (alive l31)
    (alive l33)
    (alive l41)
    (alive l42)
    (alive l44)
    (alive l45)
    (alive l55)
    (NEIGHBOR l11 l12)
    (NEIGHBOR l11 l21)
    (NEIGHBOR l11 l22)
    (NEIGHBOR l12 l11)
    (NEIGHBOR l12 l13)
    (NEIGHBOR l12 l21)
    (NEIGHBOR l12 l22)
    (NEIGHBOR l12 l23)
    (NEIGHBOR l13 l12)
    (NEIGHBOR l13 l14)
    (NEIGHBOR l13 l22)
    (NEIGHBOR l13 l23)
    (NEIGHBOR l13 l24)
    (NEIGHBOR l14 l13)
    (NEIGHBOR l14 l15)
    (NEIGHBOR l14 l23)
    (NEIGHBOR l14 l24)
    (NEIGHBOR l14 l25)
    (NEIGHBOR l15 l14)
    (NEIGHBOR l15 l24)
    (NEIGHBOR l15 l25)
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
    (NEIGHBOR l23 l14)
    (NEIGHBOR l23 l22)
    (NEIGHBOR l23 l24)
    (NEIGHBOR l23 l32)
    (NEIGHBOR l23 l33)
    (NEIGHBOR l23 l34)
    (NEIGHBOR l24 l13)
    (NEIGHBOR l24 l14)
    (NEIGHBOR l24 l15)
    (NEIGHBOR l24 l23)
    (NEIGHBOR l24 l25)
    (NEIGHBOR l24 l33)
    (NEIGHBOR l24 l34)
    (NEIGHBOR l24 l35)
    (NEIGHBOR l25 l14)
    (NEIGHBOR l25 l15)
    (NEIGHBOR l25 l24)
    (NEIGHBOR l25 l34)
    (NEIGHBOR l25 l35)
    (NEIGHBOR l31 l21)
    (NEIGHBOR l31 l22)
    (NEIGHBOR l31 l32)
    (NEIGHBOR l31 l41)
    (NEIGHBOR l31 l42)
    (NEIGHBOR l32 l21)
    (NEIGHBOR l32 l22)
    (NEIGHBOR l32 l23)
    (NEIGHBOR l32 l31)
    (NEIGHBOR l32 l33)
    (NEIGHBOR l32 l41)
    (NEIGHBOR l32 l42)
    (NEIGHBOR l32 l43)
    (NEIGHBOR l33 l22)
    (NEIGHBOR l33 l23)
    (NEIGHBOR l33 l24)
    (NEIGHBOR l33 l32)
    (NEIGHBOR l33 l34)
    (NEIGHBOR l33 l42)
    (NEIGHBOR l33 l43)
    (NEIGHBOR l33 l44)
    (NEIGHBOR l34 l23)
    (NEIGHBOR l34 l24)
    (NEIGHBOR l34 l25)
    (NEIGHBOR l34 l33)
    (NEIGHBOR l34 l35)
    (NEIGHBOR l34 l43)
    (NEIGHBOR l34 l44)
    (NEIGHBOR l34 l45)
    (NEIGHBOR l35 l24)
    (NEIGHBOR l35 l25)
    (NEIGHBOR l35 l34)
    (NEIGHBOR l35 l44)
    (NEIGHBOR l35 l45)
    (NEIGHBOR l41 l31)
    (NEIGHBOR l41 l32)
    (NEIGHBOR l41 l42)
    (NEIGHBOR l41 l51)
    (NEIGHBOR l41 l52)
    (NEIGHBOR l42 l31)
    (NEIGHBOR l42 l32)
    (NEIGHBOR l42 l33)
    (NEIGHBOR l42 l41)
    (NEIGHBOR l42 l43)
    (NEIGHBOR l42 l51)
    (NEIGHBOR l42 l52)
    (NEIGHBOR l42 l53)
    (NEIGHBOR l43 l32)
    (NEIGHBOR l43 l33)
    (NEIGHBOR l43 l34)
    (NEIGHBOR l43 l42)
    (NEIGHBOR l43 l44)
    (NEIGHBOR l43 l52)
    (NEIGHBOR l43 l53)
    (NEIGHBOR l43 l54)
    (NEIGHBOR l44 l33)
    (NEIGHBOR l44 l34)
    (NEIGHBOR l44 l35)
    (NEIGHBOR l44 l43)
    (NEIGHBOR l44 l45)
    (NEIGHBOR l44 l53)
    (NEIGHBOR l44 l54)
    (NEIGHBOR l44 l55)
    (NEIGHBOR l45 l34)
    (NEIGHBOR l45 l35)
    (NEIGHBOR l45 l44)
    (NEIGHBOR l45 l54)
    (NEIGHBOR l45 l55)
    (NEIGHBOR l51 l41)
    (NEIGHBOR l51 l42)
    (NEIGHBOR l51 l52)
    (NEIGHBOR l52 l41)
    (NEIGHBOR l52 l42)
    (NEIGHBOR l52 l43)
    (NEIGHBOR l52 l51)
    (NEIGHBOR l52 l53)
    (NEIGHBOR l53 l42)
    (NEIGHBOR l53 l43)
    (NEIGHBOR l53 l44)
    (NEIGHBOR l53 l52)
    (NEIGHBOR l53 l54)
    (NEIGHBOR l54 l43)
    (NEIGHBOR l54 l44)
    (NEIGHBOR l54 l45)
    (NEIGHBOR l54 l53)
    (NEIGHBOR l54 l55)
    (NEIGHBOR l55 l44)
    (NEIGHBOR l55 l45)
    (NEIGHBOR l55 l54))
(:goal (and)))
; <magic_json> {"rddl": true}

