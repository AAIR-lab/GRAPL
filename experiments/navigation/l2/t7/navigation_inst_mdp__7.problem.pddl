(define (problem navigation_inst_mdp__7)
(:domain navigation_mdp)
(:objects
    y36 - ypos
    y27 - ypos
    y15 - ypos
    y12 - ypos
    y20 - ypos
    x9 - xpos
    x21 - xpos
    x54 - xpos
    x69 - xpos
    x30 - xpos
    x6 - xpos
    x14 - xpos
    x41 - xpos
    x86 - xpos
    x105 - xpos)
(:init
    (robot-at x105 y12)
    (EAST x30 x41)
    (SOUTH y36 y27)
    (GOAL x105 y36)
    (SOUTH y20 y15)
    (WEST x30 x21)
    (EAST x21 x30)
    (NORTH y27 y36)
    (WEST x69 x54)
    (SOUTH y15 y12)
    (MIN-XPOS x6)
    (MAX-XPOS x105)
    (WEST x21 x14)
    (EAST x9 x14)
    (NORTH y15 y20)
    (WEST x105 x86)
    (WEST x41 x30)
    (EAST x86 x105)
    (WEST x9 x6)
    (EAST x41 x54)
    (WEST x54 x41)
    (MAX-YPOS y36)
    (EAST x54 x69)
    (EAST x14 x21)
    (SOUTH y27 y20)
    (EAST x6 x9)
    (EAST x69 x86)
    (WEST x14 x9)
    (WEST x86 x69)
    (NORTH y20 y27)
    (NORTH y12 y15)
    (MIN-YPOS y12))
(:goal (and)))
; <magic_json> {"rddl": true}

