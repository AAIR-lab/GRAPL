(define (problem navigation_inst_mdp__6)
(:domain navigation_mdp)
(:objects
    y12 - ypos
    y15 - ypos
    y27 - ypos
    y20 - ypos
    x54 - xpos
    x9 - xpos
    x86 - xpos
    x30 - xpos
    x105 - xpos
    x14 - xpos
    x69 - xpos
    x6 - xpos
    x41 - xpos
    x21 - xpos)
(:init
    (robot-at x105 y12)
    (EAST x30 x41)
    (EAST x21 x30)
    (MAX-XPOS x105)
    (WEST x86 x69)
    (WEST x30 x21)
    (WEST x14 x9)
    (NORTH y20 y27)
    (SOUTH y27 y20)
    (NORTH y15 y20)
    (WEST x54 x41)
    (WEST x69 x54)
    (SOUTH y20 y15)
    (WEST x21 x14)
    (EAST x14 x21)
    (WEST x9 x6)
    (MAX-YPOS y27)
    (EAST x86 x105)
    (SOUTH y15 y12)
    (MIN-XPOS x6)
    (EAST x6 x9)
    (EAST x41 x54)
    (EAST x9 x14)
    (GOAL x105 y27)
    (NORTH y12 y15)
    (MIN-YPOS y12)
    (EAST x69 x86)
    (WEST x105 x86)
    (EAST x54 x69)
    (WEST x41 x30))
(:goal (and)))
; <magic_json> {"rddl": true}

