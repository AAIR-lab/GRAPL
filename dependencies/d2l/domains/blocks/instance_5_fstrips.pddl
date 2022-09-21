
(define (problem instance_5_1)
  (:domain blocksworld-fn)
  (:objects
    b1 b2 b3 b4 b5 - block
  )

  (:init
    (= (loc b1) b4)
	(= (loc b2) table)
	(= (loc b3) table)
	(= (loc b4) table)
	(= (loc b5) table)
	(clear b1)
	(clear b2)
	(clear b3)
	(clear b5)
	(clear table)
  )

  (:goal
    (and 
	(= (loc b1) b4)
	(= (loc b2) table)
	(= (loc b3) table)
	(= (loc b4) table)
	(= (loc b5) b2)
	)
  )

  

  

  

  
)
