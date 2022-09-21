
(define (problem instance_5_1)
  (:domain blocksworld-fn)
  (:objects
    a b2 b3 b4 b5 - block
  )

  (:init
    (= (loc a) table)
	(= (loc b2) table)
	(= (loc b3) table)
	(= (loc b4) table)
	(= (loc b5) table)
	(clear a)
	(clear b2)
	(clear b3)
	(clear b4)
	(clear b5)
	(clear table)
  )

  (:goal
    (and 
	(clear a)
	)
  )

  

  

  

  
)
