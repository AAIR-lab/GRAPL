(define (problem navigation_inst_mdp__5)
(:domain navigation_mdp)
(:objects
    y12 - ypos
    y15 - ypos
    y20 - ypos
    x14 - xpos
    x54 - xpos
    x6 - xpos
    x86 - xpos
    x41 - xpos
    x9 - xpos
    x21 - xpos
    x30 - xpos
    x69 - xpos
    x105 - xpos)
(:init
    (robot-at x105 y12)
    (MIN-YPOS y12)
    (SOUTH y15 y12)
    (EAST x9 x14)
    (WEST x41 x30)
    (WEST x21 x14)
    (EAST x69 x86)
    (WEST x14 x9)
    (EAST x21 x30)
    (WEST x69 x54)
    (GOAL x105 y20)
    (MAX-XPOS x105)
    (EAST x54 x69)
    (EAST x30 x41)
    (WEST x30 x21)
    (MAX-YPOS y20)
    (WEST x105 x86)
    (NORTH y15 y20)
    (MIN-XPOS x6)
    (NORTH y12 y15)
    (WEST x9 x6)
    (WEST x86 x69)
    (EAST x86 x105)
    (WEST x54 x41)
    (EAST x14 x21)
    (EAST x6 x9)
    (EAST x41 x54)
    (SOUTH y20 y15))
(:goal (and)))
; <magic_json> {"rddl": true}

