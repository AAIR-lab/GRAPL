(define (problem sample)
  (:domain fn-blocksworld-tower)
  (:objects
    b1 b2 b3 b4 - block
  )

  (:init
    (clear b1)
	(clear b2)
	(clear b3)
	(= (loc b1) table)
	(= (loc b2) b4)
	(= (loc b3) table)
	(= (loc b4) table)
	(clear table)
  )

  (:goal (and
  	(@alldiff (loc b1) (loc b2) (loc b3) (loc b4))
  ))
)
