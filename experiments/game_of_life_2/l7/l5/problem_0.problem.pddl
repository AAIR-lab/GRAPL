(define (problem problem_0)
(:domain game_of_life_2_mdp)
(:objects
    l11 - location
    l12 - location
    l13 - location
    l14 - location
    l15 - location
    l16 - location
    l17 - location
    l18 - location
    l19 - location
    l21 - location
    l22 - location
    l23 - location
    l24 - location
    l25 - location
    l26 - location
    l27 - location
    l28 - location
    l29 - location
    l31 - location
    l32 - location
    l33 - location
    l34 - location
    l35 - location
    l36 - location
    l37 - location
    l38 - location
    l39 - location
    l41 - location
    l42 - location
    l43 - location
    l44 - location
    l45 - location
    l46 - location
    l47 - location
    l48 - location
    l49 - location
    l51 - location
    l52 - location
    l53 - location
    l54 - location
    l55 - location
    l56 - location
    l57 - location
    l58 - location
    l59 - location
    l61 - location
    l62 - location
    l63 - location
    l64 - location
    l65 - location
    l66 - location
    l67 - location
    l68 - location
    l69 - location
    l71 - location
    l72 - location
    l73 - location
    l74 - location
    l75 - location
    l76 - location
    l77 - location
    l78 - location
    l79 - location
    l81 - location
    l82 - location
    l83 - location
    l84 - location
    l85 - location
    l86 - location
    l87 - location
    l88 - location
    l89 - location
    l91 - location
    l92 - location
    l93 - location
    l94 - location
    l95 - location
    l96 - location
    l97 - location
    l98 - location
    l99 - location)
(:init
    (alive l11)
    (alive l12)
    (alive l15)
    (alive l16)
    (alive l18)
    (alive l19)
    (alive l21)
    (alive l22)
    (alive l23)
    (alive l24)
    (alive l25)
    (alive l26)
    (alive l27)
    (alive l29)
    (alive l31)
    (alive l33)
    (alive l34)
    (alive l36)
    (alive l38)
    (alive l42)
    (alive l44)
    (alive l45)
    (alive l46)
    (alive l52)
    (alive l54)
    (alive l55)
    (alive l61)
    (alive l63)
    (alive l65)
    (alive l66)
    (alive l67)
    (alive l68)
    (alive l71)
    (alive l72)
    (alive l74)
    (alive l78)
    (alive l79)
    (alive l82)
    (alive l83)
    (alive l84)
    (alive l86)
    (alive l87)
    (alive l89)
    (alive l93)
    (alive l94)
    (alive l96)
    (alive l97)
    (alive l98)
    (alive l99)
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
    (NEIGHBOR l15 l16)
    (NEIGHBOR l15 l24)
    (NEIGHBOR l15 l25)
    (NEIGHBOR l15 l26)
    (NEIGHBOR l16 l15)
    (NEIGHBOR l16 l17)
    (NEIGHBOR l16 l25)
    (NEIGHBOR l16 l26)
    (NEIGHBOR l16 l27)
    (NEIGHBOR l17 l16)
    (NEIGHBOR l17 l18)
    (NEIGHBOR l17 l26)
    (NEIGHBOR l17 l27)
    (NEIGHBOR l17 l28)
    (NEIGHBOR l18 l17)
    (NEIGHBOR l18 l19)
    (NEIGHBOR l18 l27)
    (NEIGHBOR l18 l28)
    (NEIGHBOR l18 l29)
    (NEIGHBOR l19 l18)
    (NEIGHBOR l19 l28)
    (NEIGHBOR l19 l29)
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
    (NEIGHBOR l25 l16)
    (NEIGHBOR l25 l24)
    (NEIGHBOR l25 l26)
    (NEIGHBOR l25 l34)
    (NEIGHBOR l25 l35)
    (NEIGHBOR l25 l36)
    (NEIGHBOR l26 l15)
    (NEIGHBOR l26 l16)
    (NEIGHBOR l26 l17)
    (NEIGHBOR l26 l25)
    (NEIGHBOR l26 l27)
    (NEIGHBOR l26 l35)
    (NEIGHBOR l26 l36)
    (NEIGHBOR l26 l37)
    (NEIGHBOR l27 l16)
    (NEIGHBOR l27 l17)
    (NEIGHBOR l27 l18)
    (NEIGHBOR l27 l26)
    (NEIGHBOR l27 l28)
    (NEIGHBOR l27 l36)
    (NEIGHBOR l27 l37)
    (NEIGHBOR l27 l38)
    (NEIGHBOR l28 l17)
    (NEIGHBOR l28 l18)
    (NEIGHBOR l28 l19)
    (NEIGHBOR l28 l27)
    (NEIGHBOR l28 l29)
    (NEIGHBOR l28 l37)
    (NEIGHBOR l28 l38)
    (NEIGHBOR l28 l39)
    (NEIGHBOR l29 l18)
    (NEIGHBOR l29 l19)
    (NEIGHBOR l29 l28)
    (NEIGHBOR l29 l38)
    (NEIGHBOR l29 l39)
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
    (NEIGHBOR l35 l26)
    (NEIGHBOR l35 l34)
    (NEIGHBOR l35 l36)
    (NEIGHBOR l35 l44)
    (NEIGHBOR l35 l45)
    (NEIGHBOR l35 l46)
    (NEIGHBOR l36 l25)
    (NEIGHBOR l36 l26)
    (NEIGHBOR l36 l27)
    (NEIGHBOR l36 l35)
    (NEIGHBOR l36 l37)
    (NEIGHBOR l36 l45)
    (NEIGHBOR l36 l46)
    (NEIGHBOR l36 l47)
    (NEIGHBOR l37 l26)
    (NEIGHBOR l37 l27)
    (NEIGHBOR l37 l28)
    (NEIGHBOR l37 l36)
    (NEIGHBOR l37 l38)
    (NEIGHBOR l37 l46)
    (NEIGHBOR l37 l47)
    (NEIGHBOR l37 l48)
    (NEIGHBOR l38 l27)
    (NEIGHBOR l38 l28)
    (NEIGHBOR l38 l29)
    (NEIGHBOR l38 l37)
    (NEIGHBOR l38 l39)
    (NEIGHBOR l38 l47)
    (NEIGHBOR l38 l48)
    (NEIGHBOR l38 l49)
    (NEIGHBOR l39 l28)
    (NEIGHBOR l39 l29)
    (NEIGHBOR l39 l38)
    (NEIGHBOR l39 l48)
    (NEIGHBOR l39 l49)
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
    (NEIGHBOR l45 l36)
    (NEIGHBOR l45 l44)
    (NEIGHBOR l45 l46)
    (NEIGHBOR l45 l54)
    (NEIGHBOR l45 l55)
    (NEIGHBOR l45 l56)
    (NEIGHBOR l46 l35)
    (NEIGHBOR l46 l36)
    (NEIGHBOR l46 l37)
    (NEIGHBOR l46 l45)
    (NEIGHBOR l46 l47)
    (NEIGHBOR l46 l55)
    (NEIGHBOR l46 l56)
    (NEIGHBOR l46 l57)
    (NEIGHBOR l47 l36)
    (NEIGHBOR l47 l37)
    (NEIGHBOR l47 l38)
    (NEIGHBOR l47 l46)
    (NEIGHBOR l47 l48)
    (NEIGHBOR l47 l56)
    (NEIGHBOR l47 l57)
    (NEIGHBOR l47 l58)
    (NEIGHBOR l48 l37)
    (NEIGHBOR l48 l38)
    (NEIGHBOR l48 l39)
    (NEIGHBOR l48 l47)
    (NEIGHBOR l48 l49)
    (NEIGHBOR l48 l57)
    (NEIGHBOR l48 l58)
    (NEIGHBOR l48 l59)
    (NEIGHBOR l49 l38)
    (NEIGHBOR l49 l39)
    (NEIGHBOR l49 l48)
    (NEIGHBOR l49 l58)
    (NEIGHBOR l49 l59)
    (NEIGHBOR l51 l41)
    (NEIGHBOR l51 l42)
    (NEIGHBOR l51 l52)
    (NEIGHBOR l51 l61)
    (NEIGHBOR l51 l62)
    (NEIGHBOR l52 l41)
    (NEIGHBOR l52 l42)
    (NEIGHBOR l52 l43)
    (NEIGHBOR l52 l51)
    (NEIGHBOR l52 l53)
    (NEIGHBOR l52 l61)
    (NEIGHBOR l52 l62)
    (NEIGHBOR l52 l63)
    (NEIGHBOR l53 l42)
    (NEIGHBOR l53 l43)
    (NEIGHBOR l53 l44)
    (NEIGHBOR l53 l52)
    (NEIGHBOR l53 l54)
    (NEIGHBOR l53 l62)
    (NEIGHBOR l53 l63)
    (NEIGHBOR l53 l64)
    (NEIGHBOR l54 l43)
    (NEIGHBOR l54 l44)
    (NEIGHBOR l54 l45)
    (NEIGHBOR l54 l53)
    (NEIGHBOR l54 l55)
    (NEIGHBOR l54 l63)
    (NEIGHBOR l54 l64)
    (NEIGHBOR l54 l65)
    (NEIGHBOR l55 l44)
    (NEIGHBOR l55 l45)
    (NEIGHBOR l55 l46)
    (NEIGHBOR l55 l54)
    (NEIGHBOR l55 l56)
    (NEIGHBOR l55 l64)
    (NEIGHBOR l55 l65)
    (NEIGHBOR l55 l66)
    (NEIGHBOR l56 l45)
    (NEIGHBOR l56 l46)
    (NEIGHBOR l56 l47)
    (NEIGHBOR l56 l55)
    (NEIGHBOR l56 l57)
    (NEIGHBOR l56 l65)
    (NEIGHBOR l56 l66)
    (NEIGHBOR l56 l67)
    (NEIGHBOR l57 l46)
    (NEIGHBOR l57 l47)
    (NEIGHBOR l57 l48)
    (NEIGHBOR l57 l56)
    (NEIGHBOR l57 l58)
    (NEIGHBOR l57 l66)
    (NEIGHBOR l57 l67)
    (NEIGHBOR l57 l68)
    (NEIGHBOR l58 l47)
    (NEIGHBOR l58 l48)
    (NEIGHBOR l58 l49)
    (NEIGHBOR l58 l57)
    (NEIGHBOR l58 l59)
    (NEIGHBOR l58 l67)
    (NEIGHBOR l58 l68)
    (NEIGHBOR l58 l69)
    (NEIGHBOR l59 l48)
    (NEIGHBOR l59 l49)
    (NEIGHBOR l59 l58)
    (NEIGHBOR l59 l68)
    (NEIGHBOR l59 l69)
    (NEIGHBOR l61 l51)
    (NEIGHBOR l61 l52)
    (NEIGHBOR l61 l62)
    (NEIGHBOR l61 l71)
    (NEIGHBOR l61 l72)
    (NEIGHBOR l62 l51)
    (NEIGHBOR l62 l52)
    (NEIGHBOR l62 l53)
    (NEIGHBOR l62 l61)
    (NEIGHBOR l62 l63)
    (NEIGHBOR l62 l71)
    (NEIGHBOR l62 l72)
    (NEIGHBOR l62 l73)
    (NEIGHBOR l63 l52)
    (NEIGHBOR l63 l53)
    (NEIGHBOR l63 l54)
    (NEIGHBOR l63 l62)
    (NEIGHBOR l63 l64)
    (NEIGHBOR l63 l72)
    (NEIGHBOR l63 l73)
    (NEIGHBOR l63 l74)
    (NEIGHBOR l64 l53)
    (NEIGHBOR l64 l54)
    (NEIGHBOR l64 l55)
    (NEIGHBOR l64 l63)
    (NEIGHBOR l64 l65)
    (NEIGHBOR l64 l73)
    (NEIGHBOR l64 l74)
    (NEIGHBOR l64 l75)
    (NEIGHBOR l65 l54)
    (NEIGHBOR l65 l55)
    (NEIGHBOR l65 l56)
    (NEIGHBOR l65 l64)
    (NEIGHBOR l65 l66)
    (NEIGHBOR l65 l74)
    (NEIGHBOR l65 l75)
    (NEIGHBOR l65 l76)
    (NEIGHBOR l66 l55)
    (NEIGHBOR l66 l56)
    (NEIGHBOR l66 l57)
    (NEIGHBOR l66 l65)
    (NEIGHBOR l66 l67)
    (NEIGHBOR l66 l75)
    (NEIGHBOR l66 l76)
    (NEIGHBOR l66 l77)
    (NEIGHBOR l67 l56)
    (NEIGHBOR l67 l57)
    (NEIGHBOR l67 l58)
    (NEIGHBOR l67 l66)
    (NEIGHBOR l67 l68)
    (NEIGHBOR l67 l76)
    (NEIGHBOR l67 l77)
    (NEIGHBOR l67 l78)
    (NEIGHBOR l68 l57)
    (NEIGHBOR l68 l58)
    (NEIGHBOR l68 l59)
    (NEIGHBOR l68 l67)
    (NEIGHBOR l68 l69)
    (NEIGHBOR l68 l77)
    (NEIGHBOR l68 l78)
    (NEIGHBOR l68 l79)
    (NEIGHBOR l69 l58)
    (NEIGHBOR l69 l59)
    (NEIGHBOR l69 l68)
    (NEIGHBOR l69 l78)
    (NEIGHBOR l69 l79)
    (NEIGHBOR l71 l61)
    (NEIGHBOR l71 l62)
    (NEIGHBOR l71 l72)
    (NEIGHBOR l71 l81)
    (NEIGHBOR l71 l82)
    (NEIGHBOR l72 l61)
    (NEIGHBOR l72 l62)
    (NEIGHBOR l72 l63)
    (NEIGHBOR l72 l71)
    (NEIGHBOR l72 l73)
    (NEIGHBOR l72 l81)
    (NEIGHBOR l72 l82)
    (NEIGHBOR l72 l83)
    (NEIGHBOR l73 l62)
    (NEIGHBOR l73 l63)
    (NEIGHBOR l73 l64)
    (NEIGHBOR l73 l72)
    (NEIGHBOR l73 l74)
    (NEIGHBOR l73 l82)
    (NEIGHBOR l73 l83)
    (NEIGHBOR l73 l84)
    (NEIGHBOR l74 l63)
    (NEIGHBOR l74 l64)
    (NEIGHBOR l74 l65)
    (NEIGHBOR l74 l73)
    (NEIGHBOR l74 l75)
    (NEIGHBOR l74 l83)
    (NEIGHBOR l74 l84)
    (NEIGHBOR l74 l85)
    (NEIGHBOR l75 l64)
    (NEIGHBOR l75 l65)
    (NEIGHBOR l75 l66)
    (NEIGHBOR l75 l74)
    (NEIGHBOR l75 l76)
    (NEIGHBOR l75 l84)
    (NEIGHBOR l75 l85)
    (NEIGHBOR l75 l86)
    (NEIGHBOR l76 l65)
    (NEIGHBOR l76 l66)
    (NEIGHBOR l76 l67)
    (NEIGHBOR l76 l75)
    (NEIGHBOR l76 l77)
    (NEIGHBOR l76 l85)
    (NEIGHBOR l76 l86)
    (NEIGHBOR l76 l87)
    (NEIGHBOR l77 l66)
    (NEIGHBOR l77 l67)
    (NEIGHBOR l77 l68)
    (NEIGHBOR l77 l76)
    (NEIGHBOR l77 l78)
    (NEIGHBOR l77 l86)
    (NEIGHBOR l77 l87)
    (NEIGHBOR l77 l88)
    (NEIGHBOR l78 l67)
    (NEIGHBOR l78 l68)
    (NEIGHBOR l78 l69)
    (NEIGHBOR l78 l77)
    (NEIGHBOR l78 l79)
    (NEIGHBOR l78 l87)
    (NEIGHBOR l78 l88)
    (NEIGHBOR l78 l89)
    (NEIGHBOR l79 l68)
    (NEIGHBOR l79 l69)
    (NEIGHBOR l79 l78)
    (NEIGHBOR l79 l88)
    (NEIGHBOR l79 l89)
    (NEIGHBOR l81 l71)
    (NEIGHBOR l81 l72)
    (NEIGHBOR l81 l82)
    (NEIGHBOR l81 l91)
    (NEIGHBOR l81 l92)
    (NEIGHBOR l82 l71)
    (NEIGHBOR l82 l72)
    (NEIGHBOR l82 l73)
    (NEIGHBOR l82 l81)
    (NEIGHBOR l82 l83)
    (NEIGHBOR l82 l91)
    (NEIGHBOR l82 l92)
    (NEIGHBOR l82 l93)
    (NEIGHBOR l83 l72)
    (NEIGHBOR l83 l73)
    (NEIGHBOR l83 l74)
    (NEIGHBOR l83 l82)
    (NEIGHBOR l83 l84)
    (NEIGHBOR l83 l92)
    (NEIGHBOR l83 l93)
    (NEIGHBOR l83 l94)
    (NEIGHBOR l84 l73)
    (NEIGHBOR l84 l74)
    (NEIGHBOR l84 l75)
    (NEIGHBOR l84 l83)
    (NEIGHBOR l84 l85)
    (NEIGHBOR l84 l93)
    (NEIGHBOR l84 l94)
    (NEIGHBOR l84 l95)
    (NEIGHBOR l85 l74)
    (NEIGHBOR l85 l75)
    (NEIGHBOR l85 l76)
    (NEIGHBOR l85 l84)
    (NEIGHBOR l85 l86)
    (NEIGHBOR l85 l94)
    (NEIGHBOR l85 l95)
    (NEIGHBOR l85 l96)
    (NEIGHBOR l86 l75)
    (NEIGHBOR l86 l76)
    (NEIGHBOR l86 l77)
    (NEIGHBOR l86 l85)
    (NEIGHBOR l86 l87)
    (NEIGHBOR l86 l95)
    (NEIGHBOR l86 l96)
    (NEIGHBOR l86 l97)
    (NEIGHBOR l87 l76)
    (NEIGHBOR l87 l77)
    (NEIGHBOR l87 l78)
    (NEIGHBOR l87 l86)
    (NEIGHBOR l87 l88)
    (NEIGHBOR l87 l96)
    (NEIGHBOR l87 l97)
    (NEIGHBOR l87 l98)
    (NEIGHBOR l88 l77)
    (NEIGHBOR l88 l78)
    (NEIGHBOR l88 l79)
    (NEIGHBOR l88 l87)
    (NEIGHBOR l88 l89)
    (NEIGHBOR l88 l97)
    (NEIGHBOR l88 l98)
    (NEIGHBOR l88 l99)
    (NEIGHBOR l89 l78)
    (NEIGHBOR l89 l79)
    (NEIGHBOR l89 l88)
    (NEIGHBOR l89 l98)
    (NEIGHBOR l89 l99)
    (NEIGHBOR l91 l81)
    (NEIGHBOR l91 l82)
    (NEIGHBOR l91 l92)
    (NEIGHBOR l92 l81)
    (NEIGHBOR l92 l82)
    (NEIGHBOR l92 l83)
    (NEIGHBOR l92 l91)
    (NEIGHBOR l92 l93)
    (NEIGHBOR l93 l82)
    (NEIGHBOR l93 l83)
    (NEIGHBOR l93 l84)
    (NEIGHBOR l93 l92)
    (NEIGHBOR l93 l94)
    (NEIGHBOR l94 l83)
    (NEIGHBOR l94 l84)
    (NEIGHBOR l94 l85)
    (NEIGHBOR l94 l93)
    (NEIGHBOR l94 l95)
    (NEIGHBOR l95 l84)
    (NEIGHBOR l95 l85)
    (NEIGHBOR l95 l86)
    (NEIGHBOR l95 l94)
    (NEIGHBOR l95 l96)
    (NEIGHBOR l96 l85)
    (NEIGHBOR l96 l86)
    (NEIGHBOR l96 l87)
    (NEIGHBOR l96 l95)
    (NEIGHBOR l96 l97)
    (NEIGHBOR l97 l86)
    (NEIGHBOR l97 l87)
    (NEIGHBOR l97 l88)
    (NEIGHBOR l97 l96)
    (NEIGHBOR l97 l98)
    (NEIGHBOR l98 l87)
    (NEIGHBOR l98 l88)
    (NEIGHBOR l98 l89)
    (NEIGHBOR l98 l97)
    (NEIGHBOR l98 l99)
    (NEIGHBOR l99 l88)
    (NEIGHBOR l99 l89)
    (NEIGHBOR l99 l98))
(:goal (and)))
; <magic_json> {"rddl": true}

