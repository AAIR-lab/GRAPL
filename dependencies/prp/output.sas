begin_version
3
end_version
begin_metric
0
end_metric
10
begin_variable
var0
-1
2
Atom hasspare()
NegatedAtom hasspare()
end_variable
begin_variable
var1
-1
2
Atom not-flattire()
NegatedAtom not-flattire()
end_variable
begin_variable
var2
-1
2
Atom spare-in(n0)
NegatedAtom spare-in(n0)
end_variable
begin_variable
var3
-1
2
Atom spare-in(n1)
NegatedAtom spare-in(n1)
end_variable
begin_variable
var4
-1
2
Atom spare-in(n15)
NegatedAtom spare-in(n15)
end_variable
begin_variable
var5
-1
2
Atom spare-in(n16)
NegatedAtom spare-in(n16)
end_variable
begin_variable
var6
-1
2
Atom spare-in(n20)
NegatedAtom spare-in(n20)
end_variable
begin_variable
var7
-1
2
Atom spare-in(n5)
NegatedAtom spare-in(n5)
end_variable
begin_variable
var8
-1
2
Atom spare-in(n6)
NegatedAtom spare-in(n6)
end_variable
begin_variable
var9
-1
21
Atom vehicle-at(n0)
Atom vehicle-at(n1)
Atom vehicle-at(n10)
Atom vehicle-at(n11)
Atom vehicle-at(n12)
Atom vehicle-at(n13)
Atom vehicle-at(n14)
Atom vehicle-at(n15)
Atom vehicle-at(n16)
Atom vehicle-at(n17)
Atom vehicle-at(n18)
Atom vehicle-at(n19)
Atom vehicle-at(n2)
Atom vehicle-at(n20)
Atom vehicle-at(n3)
Atom vehicle-at(n4)
Atom vehicle-at(n5)
Atom vehicle-at(n6)
Atom vehicle-at(n7)
Atom vehicle-at(n8)
Atom vehicle-at(n9)
end_variable
1
begin_mutex_group
21
9 0
9 1
9 2
9 3
9 4
9 5
9 6
9 7
9 8
9 9
9 10
9 11
9 12
9 13
9 14
9 15
9 16
9 17
9 18
9 19
9 20
end_mutex_group
begin_state
1
0
0
0
0
0
0
0
0
0
end_state
begin_goal
1
9 6
end_goal
224
begin_operator
changetire_DETDUP_1 
0
2
0 0 0 1
0 1 -1 0
0
end_operator
begin_operator
loadtire n0
1
9 0
2
0 0 -1 0
0 2 0 1
0
end_operator
begin_operator
loadtire n1
1
9 1
2
0 0 -1 0
0 3 0 1
0
end_operator
begin_operator
loadtire n15
1
9 7
2
0 0 -1 0
0 4 0 1
0
end_operator
begin_operator
loadtire n16
1
9 8
2
0 0 -1 0
0 5 0 1
0
end_operator
begin_operator
loadtire n20
1
9 13
2
0 0 -1 0
0 6 0 1
0
end_operator
begin_operator
loadtire n5
1
9 16
2
0 0 -1 0
0 7 0 1
0
end_operator
begin_operator
loadtire n6
1
9 17
2
0 0 -1 0
0 8 0 1
0
end_operator
begin_operator
move-car_DETDUP_0 n0 n18
1
1 0
1
0 9 0 10
0
end_operator
begin_operator
move-car_DETDUP_0 n1 n16
1
1 0
1
0 9 1 8
0
end_operator
begin_operator
move-car_DETDUP_0 n1 n2
1
1 0
1
0 9 1 12
0
end_operator
begin_operator
move-car_DETDUP_0 n1 n7
1
1 0
1
0 9 1 18
0
end_operator
begin_operator
move-car_DETDUP_0 n10 n15
1
1 0
1
0 9 2 7
0
end_operator
begin_operator
move-car_DETDUP_0 n10 n9
1
1 0
1
0 9 2 20
0
end_operator
begin_operator
move-car_DETDUP_0 n11 n14
1
1 0
1
0 9 3 6
0
end_operator
begin_operator
move-car_DETDUP_0 n12 n13
1
1 0
1
0 9 4 5
0
end_operator
begin_operator
move-car_DETDUP_0 n12 n16
1
1 0
1
0 9 4 8
0
end_operator
begin_operator
move-car_DETDUP_0 n12 n4
1
1 0
1
0 9 4 15
0
end_operator
begin_operator
move-car_DETDUP_0 n13 n12
1
1 0
1
0 9 5 4
0
end_operator
begin_operator
move-car_DETDUP_0 n13 n2
1
1 0
1
0 9 5 12
0
end_operator
begin_operator
move-car_DETDUP_0 n13 n6
1
1 0
1
0 9 5 17
0
end_operator
begin_operator
move-car_DETDUP_0 n13 n8
1
1 0
1
0 9 5 19
0
end_operator
begin_operator
move-car_DETDUP_0 n14 n11
1
1 0
1
0 9 6 3
0
end_operator
begin_operator
move-car_DETDUP_0 n14 n15
1
1 0
1
0 9 6 7
0
end_operator
begin_operator
move-car_DETDUP_0 n14 n18
1
1 0
1
0 9 6 10
0
end_operator
begin_operator
move-car_DETDUP_0 n14 n3
1
1 0
1
0 9 6 14
0
end_operator
begin_operator
move-car_DETDUP_0 n14 n8
1
1 0
1
0 9 6 19
0
end_operator
begin_operator
move-car_DETDUP_0 n15 n10
1
1 0
1
0 9 7 2
0
end_operator
begin_operator
move-car_DETDUP_0 n15 n14
1
1 0
1
0 9 7 6
0
end_operator
begin_operator
move-car_DETDUP_0 n15 n5
1
1 0
1
0 9 7 16
0
end_operator
begin_operator
move-car_DETDUP_0 n16 n1
1
1 0
1
0 9 8 1
0
end_operator
begin_operator
move-car_DETDUP_0 n16 n12
1
1 0
1
0 9 8 4
0
end_operator
begin_operator
move-car_DETDUP_0 n16 n17
1
1 0
1
0 9 8 9
0
end_operator
begin_operator
move-car_DETDUP_0 n16 n20
1
1 0
1
0 9 8 13
0
end_operator
begin_operator
move-car_DETDUP_0 n16 n4
1
1 0
1
0 9 8 15
0
end_operator
begin_operator
move-car_DETDUP_0 n16 n6
1
1 0
1
0 9 8 17
0
end_operator
begin_operator
move-car_DETDUP_0 n17 n16
1
1 0
1
0 9 9 8
0
end_operator
begin_operator
move-car_DETDUP_0 n17 n2
1
1 0
1
0 9 9 12
0
end_operator
begin_operator
move-car_DETDUP_0 n17 n20
1
1 0
1
0 9 9 13
0
end_operator
begin_operator
move-car_DETDUP_0 n17 n8
1
1 0
1
0 9 9 19
0
end_operator
begin_operator
move-car_DETDUP_0 n18 n0
1
1 0
1
0 9 10 0
0
end_operator
begin_operator
move-car_DETDUP_0 n18 n14
1
1 0
1
0 9 10 6
0
end_operator
begin_operator
move-car_DETDUP_0 n18 n19
1
1 0
1
0 9 10 11
0
end_operator
begin_operator
move-car_DETDUP_0 n18 n20
1
1 0
1
0 9 10 13
0
end_operator
begin_operator
move-car_DETDUP_0 n18 n6
1
1 0
1
0 9 10 17
0
end_operator
begin_operator
move-car_DETDUP_0 n18 n7
1
1 0
1
0 9 10 18
0
end_operator
begin_operator
move-car_DETDUP_0 n19 n18
1
1 0
1
0 9 11 10
0
end_operator
begin_operator
move-car_DETDUP_0 n19 n20
1
1 0
1
0 9 11 13
0
end_operator
begin_operator
move-car_DETDUP_0 n19 n7
1
1 0
1
0 9 11 18
0
end_operator
begin_operator
move-car_DETDUP_0 n2 n1
1
1 0
1
0 9 12 1
0
end_operator
begin_operator
move-car_DETDUP_0 n2 n13
1
1 0
1
0 9 12 5
0
end_operator
begin_operator
move-car_DETDUP_0 n2 n17
1
1 0
1
0 9 12 9
0
end_operator
begin_operator
move-car_DETDUP_0 n2 n20
1
1 0
1
0 9 12 13
0
end_operator
begin_operator
move-car_DETDUP_0 n2 n6
1
1 0
1
0 9 12 17
0
end_operator
begin_operator
move-car_DETDUP_0 n2 n8
1
1 0
1
0 9 12 19
0
end_operator
begin_operator
move-car_DETDUP_0 n20 n16
1
1 0
1
0 9 13 8
0
end_operator
begin_operator
move-car_DETDUP_0 n20 n17
1
1 0
1
0 9 13 9
0
end_operator
begin_operator
move-car_DETDUP_0 n20 n18
1
1 0
1
0 9 13 10
0
end_operator
begin_operator
move-car_DETDUP_0 n20 n19
1
1 0
1
0 9 13 11
0
end_operator
begin_operator
move-car_DETDUP_0 n20 n2
1
1 0
1
0 9 13 12
0
end_operator
begin_operator
move-car_DETDUP_0 n3 n14
1
1 0
1
0 9 14 6
0
end_operator
begin_operator
move-car_DETDUP_0 n3 n8
1
1 0
1
0 9 14 19
0
end_operator
begin_operator
move-car_DETDUP_0 n4 n12
1
1 0
1
0 9 15 4
0
end_operator
begin_operator
move-car_DETDUP_0 n4 n16
1
1 0
1
0 9 15 8
0
end_operator
begin_operator
move-car_DETDUP_0 n5 n15
1
1 0
1
0 9 16 7
0
end_operator
begin_operator
move-car_DETDUP_0 n6 n13
1
1 0
1
0 9 17 5
0
end_operator
begin_operator
move-car_DETDUP_0 n6 n16
1
1 0
1
0 9 17 8
0
end_operator
begin_operator
move-car_DETDUP_0 n6 n18
1
1 0
1
0 9 17 10
0
end_operator
begin_operator
move-car_DETDUP_0 n6 n2
1
1 0
1
0 9 17 12
0
end_operator
begin_operator
move-car_DETDUP_0 n6 n9
1
1 0
1
0 9 17 20
0
end_operator
begin_operator
move-car_DETDUP_0 n7 n1
1
1 0
1
0 9 18 1
0
end_operator
begin_operator
move-car_DETDUP_0 n7 n18
1
1 0
1
0 9 18 10
0
end_operator
begin_operator
move-car_DETDUP_0 n7 n19
1
1 0
1
0 9 18 11
0
end_operator
begin_operator
move-car_DETDUP_0 n8 n13
1
1 0
1
0 9 19 5
0
end_operator
begin_operator
move-car_DETDUP_0 n8 n14
1
1 0
1
0 9 19 6
0
end_operator
begin_operator
move-car_DETDUP_0 n8 n17
1
1 0
1
0 9 19 9
0
end_operator
begin_operator
move-car_DETDUP_0 n8 n2
1
1 0
1
0 9 19 12
0
end_operator
begin_operator
move-car_DETDUP_0 n8 n3
1
1 0
1
0 9 19 14
0
end_operator
begin_operator
move-car_DETDUP_0 n9 n10
1
1 0
1
0 9 20 2
0
end_operator
begin_operator
move-car_DETDUP_0 n9 n6
1
1 0
1
0 9 20 17
0
end_operator
begin_operator
move-car_DETDUP_1 n0 n18
1
1 0
1
0 9 0 10
0
end_operator
begin_operator
move-car_DETDUP_1 n1 n16
1
1 0
1
0 9 1 8
0
end_operator
begin_operator
move-car_DETDUP_1 n1 n2
1
1 0
1
0 9 1 12
0
end_operator
begin_operator
move-car_DETDUP_1 n1 n7
1
1 0
1
0 9 1 18
0
end_operator
begin_operator
move-car_DETDUP_1 n10 n15
1
1 0
1
0 9 2 7
0
end_operator
begin_operator
move-car_DETDUP_1 n10 n9
1
1 0
1
0 9 2 20
0
end_operator
begin_operator
move-car_DETDUP_1 n11 n14
1
1 0
1
0 9 3 6
0
end_operator
begin_operator
move-car_DETDUP_1 n12 n13
1
1 0
1
0 9 4 5
0
end_operator
begin_operator
move-car_DETDUP_1 n12 n16
1
1 0
1
0 9 4 8
0
end_operator
begin_operator
move-car_DETDUP_1 n12 n4
1
1 0
1
0 9 4 15
0
end_operator
begin_operator
move-car_DETDUP_1 n13 n12
1
1 0
1
0 9 5 4
0
end_operator
begin_operator
move-car_DETDUP_1 n13 n2
1
1 0
1
0 9 5 12
0
end_operator
begin_operator
move-car_DETDUP_1 n13 n6
1
1 0
1
0 9 5 17
0
end_operator
begin_operator
move-car_DETDUP_1 n13 n8
1
1 0
1
0 9 5 19
0
end_operator
begin_operator
move-car_DETDUP_1 n14 n11
1
1 0
1
0 9 6 3
0
end_operator
begin_operator
move-car_DETDUP_1 n14 n15
1
1 0
1
0 9 6 7
0
end_operator
begin_operator
move-car_DETDUP_1 n14 n18
1
1 0
1
0 9 6 10
0
end_operator
begin_operator
move-car_DETDUP_1 n14 n3
1
1 0
1
0 9 6 14
0
end_operator
begin_operator
move-car_DETDUP_1 n14 n8
1
1 0
1
0 9 6 19
0
end_operator
begin_operator
move-car_DETDUP_1 n15 n10
1
1 0
1
0 9 7 2
0
end_operator
begin_operator
move-car_DETDUP_1 n15 n14
1
1 0
1
0 9 7 6
0
end_operator
begin_operator
move-car_DETDUP_1 n15 n5
1
1 0
1
0 9 7 16
0
end_operator
begin_operator
move-car_DETDUP_1 n16 n1
1
1 0
1
0 9 8 1
0
end_operator
begin_operator
move-car_DETDUP_1 n16 n12
1
1 0
1
0 9 8 4
0
end_operator
begin_operator
move-car_DETDUP_1 n16 n17
1
1 0
1
0 9 8 9
0
end_operator
begin_operator
move-car_DETDUP_1 n16 n20
1
1 0
1
0 9 8 13
0
end_operator
begin_operator
move-car_DETDUP_1 n16 n4
1
1 0
1
0 9 8 15
0
end_operator
begin_operator
move-car_DETDUP_1 n16 n6
1
1 0
1
0 9 8 17
0
end_operator
begin_operator
move-car_DETDUP_1 n17 n16
1
1 0
1
0 9 9 8
0
end_operator
begin_operator
move-car_DETDUP_1 n17 n2
1
1 0
1
0 9 9 12
0
end_operator
begin_operator
move-car_DETDUP_1 n17 n20
1
1 0
1
0 9 9 13
0
end_operator
begin_operator
move-car_DETDUP_1 n17 n8
1
1 0
1
0 9 9 19
0
end_operator
begin_operator
move-car_DETDUP_1 n18 n0
1
1 0
1
0 9 10 0
0
end_operator
begin_operator
move-car_DETDUP_1 n18 n14
1
1 0
1
0 9 10 6
0
end_operator
begin_operator
move-car_DETDUP_1 n18 n19
1
1 0
1
0 9 10 11
0
end_operator
begin_operator
move-car_DETDUP_1 n18 n20
1
1 0
1
0 9 10 13
0
end_operator
begin_operator
move-car_DETDUP_1 n18 n6
1
1 0
1
0 9 10 17
0
end_operator
begin_operator
move-car_DETDUP_1 n18 n7
1
1 0
1
0 9 10 18
0
end_operator
begin_operator
move-car_DETDUP_1 n19 n18
1
1 0
1
0 9 11 10
0
end_operator
begin_operator
move-car_DETDUP_1 n19 n20
1
1 0
1
0 9 11 13
0
end_operator
begin_operator
move-car_DETDUP_1 n19 n7
1
1 0
1
0 9 11 18
0
end_operator
begin_operator
move-car_DETDUP_1 n2 n1
1
1 0
1
0 9 12 1
0
end_operator
begin_operator
move-car_DETDUP_1 n2 n13
1
1 0
1
0 9 12 5
0
end_operator
begin_operator
move-car_DETDUP_1 n2 n17
1
1 0
1
0 9 12 9
0
end_operator
begin_operator
move-car_DETDUP_1 n2 n20
1
1 0
1
0 9 12 13
0
end_operator
begin_operator
move-car_DETDUP_1 n2 n6
1
1 0
1
0 9 12 17
0
end_operator
begin_operator
move-car_DETDUP_1 n2 n8
1
1 0
1
0 9 12 19
0
end_operator
begin_operator
move-car_DETDUP_1 n20 n16
1
1 0
1
0 9 13 8
0
end_operator
begin_operator
move-car_DETDUP_1 n20 n17
1
1 0
1
0 9 13 9
0
end_operator
begin_operator
move-car_DETDUP_1 n20 n18
1
1 0
1
0 9 13 10
0
end_operator
begin_operator
move-car_DETDUP_1 n20 n19
1
1 0
1
0 9 13 11
0
end_operator
begin_operator
move-car_DETDUP_1 n20 n2
1
1 0
1
0 9 13 12
0
end_operator
begin_operator
move-car_DETDUP_1 n3 n14
1
1 0
1
0 9 14 6
0
end_operator
begin_operator
move-car_DETDUP_1 n3 n8
1
1 0
1
0 9 14 19
0
end_operator
begin_operator
move-car_DETDUP_1 n4 n12
1
1 0
1
0 9 15 4
0
end_operator
begin_operator
move-car_DETDUP_1 n4 n16
1
1 0
1
0 9 15 8
0
end_operator
begin_operator
move-car_DETDUP_1 n5 n15
1
1 0
1
0 9 16 7
0
end_operator
begin_operator
move-car_DETDUP_1 n6 n13
1
1 0
1
0 9 17 5
0
end_operator
begin_operator
move-car_DETDUP_1 n6 n16
1
1 0
1
0 9 17 8
0
end_operator
begin_operator
move-car_DETDUP_1 n6 n18
1
1 0
1
0 9 17 10
0
end_operator
begin_operator
move-car_DETDUP_1 n6 n2
1
1 0
1
0 9 17 12
0
end_operator
begin_operator
move-car_DETDUP_1 n6 n9
1
1 0
1
0 9 17 20
0
end_operator
begin_operator
move-car_DETDUP_1 n7 n1
1
1 0
1
0 9 18 1
0
end_operator
begin_operator
move-car_DETDUP_1 n7 n18
1
1 0
1
0 9 18 10
0
end_operator
begin_operator
move-car_DETDUP_1 n7 n19
1
1 0
1
0 9 18 11
0
end_operator
begin_operator
move-car_DETDUP_1 n8 n13
1
1 0
1
0 9 19 5
0
end_operator
begin_operator
move-car_DETDUP_1 n8 n14
1
1 0
1
0 9 19 6
0
end_operator
begin_operator
move-car_DETDUP_1 n8 n17
1
1 0
1
0 9 19 9
0
end_operator
begin_operator
move-car_DETDUP_1 n8 n2
1
1 0
1
0 9 19 12
0
end_operator
begin_operator
move-car_DETDUP_1 n8 n3
1
1 0
1
0 9 19 14
0
end_operator
begin_operator
move-car_DETDUP_1 n9 n10
1
1 0
1
0 9 20 2
0
end_operator
begin_operator
move-car_DETDUP_1 n9 n6
1
1 0
1
0 9 20 17
0
end_operator
begin_operator
move-car_DETDUP_2 n0 n18
0
2
0 1 0 1
0 9 0 10
0
end_operator
begin_operator
move-car_DETDUP_2 n1 n16
0
2
0 1 0 1
0 9 1 8
0
end_operator
begin_operator
move-car_DETDUP_2 n1 n2
0
2
0 1 0 1
0 9 1 12
0
end_operator
begin_operator
move-car_DETDUP_2 n1 n7
0
2
0 1 0 1
0 9 1 18
0
end_operator
begin_operator
move-car_DETDUP_2 n10 n15
0
2
0 1 0 1
0 9 2 7
0
end_operator
begin_operator
move-car_DETDUP_2 n10 n9
0
2
0 1 0 1
0 9 2 20
0
end_operator
begin_operator
move-car_DETDUP_2 n11 n14
0
2
0 1 0 1
0 9 3 6
0
end_operator
begin_operator
move-car_DETDUP_2 n12 n13
0
2
0 1 0 1
0 9 4 5
0
end_operator
begin_operator
move-car_DETDUP_2 n12 n16
0
2
0 1 0 1
0 9 4 8
0
end_operator
begin_operator
move-car_DETDUP_2 n12 n4
0
2
0 1 0 1
0 9 4 15
0
end_operator
begin_operator
move-car_DETDUP_2 n13 n12
0
2
0 1 0 1
0 9 5 4
0
end_operator
begin_operator
move-car_DETDUP_2 n13 n2
0
2
0 1 0 1
0 9 5 12
0
end_operator
begin_operator
move-car_DETDUP_2 n13 n6
0
2
0 1 0 1
0 9 5 17
0
end_operator
begin_operator
move-car_DETDUP_2 n13 n8
0
2
0 1 0 1
0 9 5 19
0
end_operator
begin_operator
move-car_DETDUP_2 n14 n11
0
2
0 1 0 1
0 9 6 3
0
end_operator
begin_operator
move-car_DETDUP_2 n14 n15
0
2
0 1 0 1
0 9 6 7
0
end_operator
begin_operator
move-car_DETDUP_2 n14 n18
0
2
0 1 0 1
0 9 6 10
0
end_operator
begin_operator
move-car_DETDUP_2 n14 n3
0
2
0 1 0 1
0 9 6 14
0
end_operator
begin_operator
move-car_DETDUP_2 n14 n8
0
2
0 1 0 1
0 9 6 19
0
end_operator
begin_operator
move-car_DETDUP_2 n15 n10
0
2
0 1 0 1
0 9 7 2
0
end_operator
begin_operator
move-car_DETDUP_2 n15 n14
0
2
0 1 0 1
0 9 7 6
0
end_operator
begin_operator
move-car_DETDUP_2 n15 n5
0
2
0 1 0 1
0 9 7 16
0
end_operator
begin_operator
move-car_DETDUP_2 n16 n1
0
2
0 1 0 1
0 9 8 1
0
end_operator
begin_operator
move-car_DETDUP_2 n16 n12
0
2
0 1 0 1
0 9 8 4
0
end_operator
begin_operator
move-car_DETDUP_2 n16 n17
0
2
0 1 0 1
0 9 8 9
0
end_operator
begin_operator
move-car_DETDUP_2 n16 n20
0
2
0 1 0 1
0 9 8 13
0
end_operator
begin_operator
move-car_DETDUP_2 n16 n4
0
2
0 1 0 1
0 9 8 15
0
end_operator
begin_operator
move-car_DETDUP_2 n16 n6
0
2
0 1 0 1
0 9 8 17
0
end_operator
begin_operator
move-car_DETDUP_2 n17 n16
0
2
0 1 0 1
0 9 9 8
0
end_operator
begin_operator
move-car_DETDUP_2 n17 n2
0
2
0 1 0 1
0 9 9 12
0
end_operator
begin_operator
move-car_DETDUP_2 n17 n20
0
2
0 1 0 1
0 9 9 13
0
end_operator
begin_operator
move-car_DETDUP_2 n17 n8
0
2
0 1 0 1
0 9 9 19
0
end_operator
begin_operator
move-car_DETDUP_2 n18 n0
0
2
0 1 0 1
0 9 10 0
0
end_operator
begin_operator
move-car_DETDUP_2 n18 n14
0
2
0 1 0 1
0 9 10 6
0
end_operator
begin_operator
move-car_DETDUP_2 n18 n19
0
2
0 1 0 1
0 9 10 11
0
end_operator
begin_operator
move-car_DETDUP_2 n18 n20
0
2
0 1 0 1
0 9 10 13
0
end_operator
begin_operator
move-car_DETDUP_2 n18 n6
0
2
0 1 0 1
0 9 10 17
0
end_operator
begin_operator
move-car_DETDUP_2 n18 n7
0
2
0 1 0 1
0 9 10 18
0
end_operator
begin_operator
move-car_DETDUP_2 n19 n18
0
2
0 1 0 1
0 9 11 10
0
end_operator
begin_operator
move-car_DETDUP_2 n19 n20
0
2
0 1 0 1
0 9 11 13
0
end_operator
begin_operator
move-car_DETDUP_2 n19 n7
0
2
0 1 0 1
0 9 11 18
0
end_operator
begin_operator
move-car_DETDUP_2 n2 n1
0
2
0 1 0 1
0 9 12 1
0
end_operator
begin_operator
move-car_DETDUP_2 n2 n13
0
2
0 1 0 1
0 9 12 5
0
end_operator
begin_operator
move-car_DETDUP_2 n2 n17
0
2
0 1 0 1
0 9 12 9
0
end_operator
begin_operator
move-car_DETDUP_2 n2 n20
0
2
0 1 0 1
0 9 12 13
0
end_operator
begin_operator
move-car_DETDUP_2 n2 n6
0
2
0 1 0 1
0 9 12 17
0
end_operator
begin_operator
move-car_DETDUP_2 n2 n8
0
2
0 1 0 1
0 9 12 19
0
end_operator
begin_operator
move-car_DETDUP_2 n20 n16
0
2
0 1 0 1
0 9 13 8
0
end_operator
begin_operator
move-car_DETDUP_2 n20 n17
0
2
0 1 0 1
0 9 13 9
0
end_operator
begin_operator
move-car_DETDUP_2 n20 n18
0
2
0 1 0 1
0 9 13 10
0
end_operator
begin_operator
move-car_DETDUP_2 n20 n19
0
2
0 1 0 1
0 9 13 11
0
end_operator
begin_operator
move-car_DETDUP_2 n20 n2
0
2
0 1 0 1
0 9 13 12
0
end_operator
begin_operator
move-car_DETDUP_2 n3 n14
0
2
0 1 0 1
0 9 14 6
0
end_operator
begin_operator
move-car_DETDUP_2 n3 n8
0
2
0 1 0 1
0 9 14 19
0
end_operator
begin_operator
move-car_DETDUP_2 n4 n12
0
2
0 1 0 1
0 9 15 4
0
end_operator
begin_operator
move-car_DETDUP_2 n4 n16
0
2
0 1 0 1
0 9 15 8
0
end_operator
begin_operator
move-car_DETDUP_2 n5 n15
0
2
0 1 0 1
0 9 16 7
0
end_operator
begin_operator
move-car_DETDUP_2 n6 n13
0
2
0 1 0 1
0 9 17 5
0
end_operator
begin_operator
move-car_DETDUP_2 n6 n16
0
2
0 1 0 1
0 9 17 8
0
end_operator
begin_operator
move-car_DETDUP_2 n6 n18
0
2
0 1 0 1
0 9 17 10
0
end_operator
begin_operator
move-car_DETDUP_2 n6 n2
0
2
0 1 0 1
0 9 17 12
0
end_operator
begin_operator
move-car_DETDUP_2 n6 n9
0
2
0 1 0 1
0 9 17 20
0
end_operator
begin_operator
move-car_DETDUP_2 n7 n1
0
2
0 1 0 1
0 9 18 1
0
end_operator
begin_operator
move-car_DETDUP_2 n7 n18
0
2
0 1 0 1
0 9 18 10
0
end_operator
begin_operator
move-car_DETDUP_2 n7 n19
0
2
0 1 0 1
0 9 18 11
0
end_operator
begin_operator
move-car_DETDUP_2 n8 n13
0
2
0 1 0 1
0 9 19 5
0
end_operator
begin_operator
move-car_DETDUP_2 n8 n14
0
2
0 1 0 1
0 9 19 6
0
end_operator
begin_operator
move-car_DETDUP_2 n8 n17
0
2
0 1 0 1
0 9 19 9
0
end_operator
begin_operator
move-car_DETDUP_2 n8 n2
0
2
0 1 0 1
0 9 19 12
0
end_operator
begin_operator
move-car_DETDUP_2 n8 n3
0
2
0 1 0 1
0 9 19 14
0
end_operator
begin_operator
move-car_DETDUP_2 n9 n10
0
2
0 1 0 1
0 9 20 2
0
end_operator
begin_operator
move-car_DETDUP_2 n9 n6
0
2
0 1 0 1
0 9 20 17
0
end_operator
0
