(define (problem sysadmin_inst_mdp__10)
(:domain sysadmin_mdp)
(:objects
    c1 - computer
    c2 - computer
    c3 - computer
    c4 - computer
    c5 - computer
    c6 - computer
    c7 - computer
    c8 - computer
    c9 - computer
    c10 - computer
    c11 - computer
    c12 - computer
    c13 - computer
    c14 - computer
    c15 - computer
    c16 - computer
    c17 - computer
    c18 - computer
    c19 - computer
    c20 - computer
    c21 - computer
    c22 - computer
    c23 - computer
    c24 - computer
    c25 - computer
    c26 - computer
    c27 - computer
    c28 - computer
    c29 - computer
    c30 - computer
    c31 - computer
    c32 - computer
    c33 - computer
    c34 - computer
    c35 - computer
    c36 - computer
    c37 - computer
    c38 - computer
    c39 - computer
    c40 - computer
    c41 - computer
    c42 - computer
    c43 - computer
    c44 - computer
    c45 - computer
    c46 - computer
    c47 - computer
    c48 - computer
    c49 - computer
    c50 - computer)
(:init
    (running c1)
    (running c2)
    (running c3)
    (running c4)
    (running c5)
    (running c6)
    (running c7)
    (running c8)
    (running c9)
    (running c10)
    (running c11)
    (running c12)
    (running c13)
    (running c14)
    (running c15)
    (running c16)
    (running c17)
    (running c18)
    (running c19)
    (running c20)
    (running c21)
    (running c22)
    (running c23)
    (running c24)
    (running c25)
    (running c26)
    (running c27)
    (running c28)
    (running c29)
    (running c30)
    (running c31)
    (running c32)
    (running c33)
    (running c34)
    (running c35)
    (running c36)
    (running c37)
    (running c38)
    (running c39)
    (running c40)
    (running c41)
    (running c42)
    (running c43)
    (running c44)
    (running c45)
    (running c46)
    (running c47)
    (running c48)
    (running c49)
    (running c50)
    (CONNECTED c1 c49)
    (CONNECTED c1 c24)
    (CONNECTED c1 c45)
    (CONNECTED c2 c16)
    (CONNECTED c2 c37)
    (CONNECTED c2 c43)
    (CONNECTED c3 c8)
    (CONNECTED c3 c27)
    (CONNECTED c3 c28)
    (CONNECTED c4 c7)
    (CONNECTED c4 c25)
    (CONNECTED c5 c25)
    (CONNECTED c5 c31)
    (CONNECTED c5 c44)
    (CONNECTED c6 c20)
    (CONNECTED c6 c37)
    (CONNECTED c6 c28)
    (CONNECTED c7 c32)
    (CONNECTED c7 c42)
    (CONNECTED c7 c28)
    (CONNECTED c8 c4)
    (CONNECTED c8 c24)
    (CONNECTED c8 c41)
    (CONNECTED c9 c22)
    (CONNECTED c9 c10)
    (CONNECTED c9 c44)
    (CONNECTED c10 c1)
    (CONNECTED c10 c21)
    (CONNECTED c10 c20)
    (CONNECTED c11 c1)
    (CONNECTED c11 c50)
    (CONNECTED c11 c21)
    (CONNECTED c12 c2)
    (CONNECTED c12 c5)
    (CONNECTED c12 c40)
    (CONNECTED c13 c2)
    (CONNECTED c13 c7)
    (CONNECTED c14 c32)
    (CONNECTED c14 c12)
    (CONNECTED c15 c16)
    (CONNECTED c15 c23)
    (CONNECTED c15 c47)
    (CONNECTED c16 c22)
    (CONNECTED c16 c42)
    (CONNECTED c16 c8)
    (CONNECTED c17 c43)
    (CONNECTED c17 c26)
    (CONNECTED c17 c28)
    (CONNECTED c18 c1)
    (CONNECTED c18 c41)
    (CONNECTED c18 c14)
    (CONNECTED c19 c48)
    (CONNECTED c19 c21)
    (CONNECTED c19 c10)
    (CONNECTED c20 c34)
    (CONNECTED c20 c28)
    (CONNECTED c20 c14)
    (CONNECTED c21 c49)
    (CONNECTED c21 c23)
    (CONNECTED c21 c24)
    (CONNECTED c22 c34)
    (CONNECTED c22 c2)
    (CONNECTED c22 c5)
    (CONNECTED c23 c49)
    (CONNECTED c23 c21)
    (CONNECTED c23 c44)
    (CONNECTED c24 c2)
    (CONNECTED c24 c36)
    (CONNECTED c24 c46)
    (CONNECTED c25 c19)
    (CONNECTED c25 c10)
    (CONNECTED c25 c31)
    (CONNECTED c26 c22)
    (CONNECTED c26 c43)
    (CONNECTED c26 c44)
    (CONNECTED c27 c19)
    (CONNECTED c27 c42)
    (CONNECTED c27 c44)
    (CONNECTED c28 c2)
    (CONNECTED c28 c19)
    (CONNECTED c28 c14)
    (CONNECTED c29 c16)
    (CONNECTED c29 c42)
    (CONNECTED c29 c27)
    (CONNECTED c30 c40)
    (CONNECTED c30 c46)
    (CONNECTED c30 c13)
    (CONNECTED c31 c35)
    (CONNECTED c31 c25)
    (CONNECTED c31 c44)
    (CONNECTED c32 c18)
    (CONNECTED c32 c22)
    (CONNECTED c32 c45)
    (CONNECTED c33 c3)
    (CONNECTED c33 c31)
    (CONNECTED c33 c15)
    (CONNECTED c34 c33)
    (CONNECTED c34 c42)
    (CONNECTED c34 c14)
    (CONNECTED c35 c39)
    (CONNECTED c35 c36)
    (CONNECTED c35 c13)
    (CONNECTED c36 c50)
    (CONNECTED c36 c8)
    (CONNECTED c36 c30)
    (CONNECTED c37 c33)
    (CONNECTED c37 c13)
    (CONNECTED c37 c15)
    (CONNECTED c38 c20)
    (CONNECTED c38 c25)
    (CONNECTED c38 c29)
    (CONNECTED c39 c1)
    (CONNECTED c39 c10)
    (CONNECTED c39 c11)
    (CONNECTED c40 c1)
    (CONNECTED c40 c18)
    (CONNECTED c40 c9)
    (CONNECTED c41 c8)
    (CONNECTED c41 c43)
    (CONNECTED c41 c13)
    (CONNECTED c42 c9)
    (CONNECTED c42 c30)
    (CONNECTED c42 c15)
    (CONNECTED c43 c34)
    (CONNECTED c43 c37)
    (CONNECTED c43 c25)
    (CONNECTED c44 c35)
    (CONNECTED c44 c43)
    (CONNECTED c44 c26)
    (CONNECTED c45 c48)
    (CONNECTED c45 c4)
    (CONNECTED c45 c41)
    (CONNECTED c46 c7)
    (CONNECTED c46 c43)
    (CONNECTED c47 c33)
    (CONNECTED c47 c20)
    (CONNECTED c47 c30)
    (CONNECTED c48 c38)
    (CONNECTED c48 c29)
    (CONNECTED c48 c44)
    (CONNECTED c49 c37)
    (CONNECTED c49 c43)
    (CONNECTED c49 c44)
    (CONNECTED c50 c34)
    (CONNECTED c50 c37)
    (CONNECTED c50 c42))
(:goal (and)))
; <magic_json> {"rddl": true}

