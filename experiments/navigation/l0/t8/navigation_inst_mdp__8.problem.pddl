(define (problem navigation_inst_mdp__8)
(:domain navigation_mdp)
(:objects
    y12 - ypos
    y20 - ypos
    y15 - ypos
    x261 - xpos
    x149 - xpos
    x21 - xpos
    x69 - xpos
    x9 - xpos
    x174 - xpos
    x41 - xpos
    x86 - xpos
    x105 - xpos
    x329 - xpos
    x14 - xpos
    x366 - xpos
    x30 - xpos
    x405 - xpos
    x201 - xpos
    x294 - xpos
    x126 - xpos
    x6 - xpos
    x230 - xpos
    x54 - xpos)
(:init
    (robot-at x405 y12)
    (WEST x14 x9)
    (EAST x30 x41)
    (WEST x149 x126)
    (WEST x41 x30)
    (EAST x69 x86)
    (GOAL x405 y20)
    (EAST x14 x21)
    (SOUTH y15 y12)
    (MAX-YPOS y20)
    (MIN-XPOS x6)
    (EAST x9 x14)
    (EAST x21 x30)
    (WEST x366 x329)
    (EAST x329 x366)
    (EAST x261 x294)
    (WEST x230 x201)
    (WEST x30 x21)
    (NORTH y15 y20)
    (WEST x54 x41)
    (EAST x174 x201)
    (EAST x201 x230)
    (MAX-XPOS x405)
    (WEST x174 x149)
    (EAST x294 x329)
    (EAST x6 x9)
    (WEST x126 x105)
    (EAST x230 x261)
    (EAST x86 x105)
    (WEST x294 x261)
    (WEST x69 x54)
    (EAST x41 x54)
    (EAST x149 x174)
    (EAST x126 x149)
    (EAST x54 x69)
    (WEST x201 x174)
    (WEST x21 x14)
    (WEST x105 x86)
    (WEST x329 x294)
    (SOUTH y20 y15)
    (WEST x261 x230)
    (WEST x86 x69)
    (NORTH y12 y15)
    (EAST x366 x405)
    (WEST x9 x6)
    (WEST x405 x366)
    (MIN-YPOS y12)
    (EAST x105 x126))
(:goal (and)))
; <magic_json> {"rddl": true}

