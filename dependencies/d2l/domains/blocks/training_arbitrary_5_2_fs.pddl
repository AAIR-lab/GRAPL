(define (problem BLOCKS-5-0)
(:domain blocksworld-fn)
(:objects a b c d e - block)

  (:init
    (= (loc a) table)
	(= (loc b) table)
	(= (loc c) table)
	(= (loc d) table)
	(= (loc e) table)
	(clear a)
	(clear b)
	(clear c)
	(clear d)
	(clear e)
	(clear table)
  )

(:goal (and
    (= (loc a) table)
    (= (loc b) table)
    (= (loc c) table)
    (= (loc d) table)
    (= (loc e) d)
))

)