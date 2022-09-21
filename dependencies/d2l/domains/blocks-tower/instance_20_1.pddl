
(define (problem instance_20_1)
  (:domain blocksworld-fn-tower)
  (:objects
    b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 - block
  )

  (:init
    (= (loc b1) b3)
	(= (loc b10) table)
	(= (loc b11) b20)
	(= (loc b12) b18)
	(= (loc b13) b12)
	(= (loc b14) table)
	(= (loc b15) b17)
	(= (loc b16) b4)
	(= (loc b17) b9)
	(= (loc b18) b15)
	(= (loc b19) b11)
	(= (loc b2) b1)
	(= (loc b20) b10)
	(= (loc b3) table)
	(= (loc b4) b8)
	(= (loc b5) b2)
	(= (loc b6) b5)
	(= (loc b7) b14)
	(= (loc b8) b13)
	(= (loc b9) b6)
	(clear b16)
	(clear b19)
	(clear b7)
	(clear table)
  )

  (:goal
    (@alldiff (loc b1) (loc b2) (loc b3) (loc b4) (loc b5) (loc b6) (loc b7) (loc b8) (loc b9) (loc b10) (loc b11) (loc b12) (loc b13) (loc b14) (loc b15) (loc b16) (loc b17) (loc b18) (loc b19) (loc b20))
  )

  

  

  

  
)
