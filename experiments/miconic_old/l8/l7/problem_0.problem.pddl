


(define (problem mixed-f14-p24-u0-v0-d0-a0-n0-A0-B0-N0-F0)
   (:domain miconic)
   (:objects p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 
             p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 
             p20 p21 p22 p23 - passenger
             f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 
             f10 f11 f12 f13 - floor)


(:init
(above f0 f1)
(above f0 f2)
(above f0 f3)
(above f0 f4)
(above f0 f5)
(above f0 f6)
(above f0 f7)
(above f0 f8)
(above f0 f9)
(above f0 f10)
(above f0 f11)
(above f0 f12)
(above f0 f13)

(above f1 f2)
(above f1 f3)
(above f1 f4)
(above f1 f5)
(above f1 f6)
(above f1 f7)
(above f1 f8)
(above f1 f9)
(above f1 f10)
(above f1 f11)
(above f1 f12)
(above f1 f13)

(above f2 f3)
(above f2 f4)
(above f2 f5)
(above f2 f6)
(above f2 f7)
(above f2 f8)
(above f2 f9)
(above f2 f10)
(above f2 f11)
(above f2 f12)
(above f2 f13)

(above f3 f4)
(above f3 f5)
(above f3 f6)
(above f3 f7)
(above f3 f8)
(above f3 f9)
(above f3 f10)
(above f3 f11)
(above f3 f12)
(above f3 f13)

(above f4 f5)
(above f4 f6)
(above f4 f7)
(above f4 f8)
(above f4 f9)
(above f4 f10)
(above f4 f11)
(above f4 f12)
(above f4 f13)

(above f5 f6)
(above f5 f7)
(above f5 f8)
(above f5 f9)
(above f5 f10)
(above f5 f11)
(above f5 f12)
(above f5 f13)

(above f6 f7)
(above f6 f8)
(above f6 f9)
(above f6 f10)
(above f6 f11)
(above f6 f12)
(above f6 f13)

(above f7 f8)
(above f7 f9)
(above f7 f10)
(above f7 f11)
(above f7 f12)
(above f7 f13)

(above f8 f9)
(above f8 f10)
(above f8 f11)
(above f8 f12)
(above f8 f13)

(above f9 f10)
(above f9 f11)
(above f9 f12)
(above f9 f13)

(above f10 f11)
(above f10 f12)
(above f10 f13)

(above f11 f12)
(above f11 f13)

(above f12 f13)



(origin p0 f11)
(destin p0 f7)

(origin p1 f8)
(destin p1 f5)

(origin p2 f8)
(destin p2 f9)

(origin p3 f8)
(destin p3 f12)

(origin p4 f11)
(destin p4 f6)

(origin p5 f7)
(destin p5 f8)

(origin p6 f12)
(destin p6 f9)

(origin p7 f3)
(destin p7 f4)

(origin p8 f10)
(destin p8 f9)

(origin p9 f9)
(destin p9 f6)

(origin p10 f5)
(destin p10 f0)

(origin p11 f13)
(destin p11 f1)

(origin p12 f8)
(destin p12 f12)

(origin p13 f10)
(destin p13 f11)

(origin p14 f7)
(destin p14 f0)

(origin p15 f8)
(destin p15 f4)

(origin p16 f7)
(destin p16 f0)

(origin p17 f10)
(destin p17 f13)

(origin p18 f10)
(destin p18 f2)

(origin p19 f11)
(destin p19 f5)

(origin p20 f9)
(destin p20 f4)

(origin p21 f12)
(destin p21 f7)

(origin p22 f12)
(destin p22 f1)

(origin p23 f9)
(destin p23 f6)






(lift-at f0)
)


(:goal (and
(served p0)
(served p1)
(served p2)
(served p3)
(served p4)
(served p5)
(served p6)
(served p7)
(served p8)
(served p9)
(served p10)
(served p11)
(served p12)
(served p13)
(served p14)
(served p15)
(served p16)
(served p17)
(served p18)
(served p19)
(served p20)
(served p21)
(served p22)
(served p23)
))
)


; <magic_json> {"min_floors": 14, "max_floors": 14, "min_passengers": 24, "max_passengers": 24, "floors": 14, "passengers": 24, "bin_params": ["floors", "passengers"]}

