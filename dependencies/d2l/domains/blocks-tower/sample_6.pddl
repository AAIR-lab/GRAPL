(define (problem sample)
  (:domain fn-blocksworld-tower)
  (:objects
    b1 b2 b3 b4 b5 b6 - block
  )

  (:init
    (clear b1)
	(clear b6)
	(clear b5)
	(= (loc b1) table)
	(= (loc b2) b4)
	(= (loc b6) b2)
	(= (loc b3) table)
	(= (loc b4) table)
	(= (loc b5) b3)
	(clear table)
  )

  (:goal (and
  	(@alldiff (loc b1) (loc b2) (loc b3) (loc b4) (loc b5))
  ))
)
