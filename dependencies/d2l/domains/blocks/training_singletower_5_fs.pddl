(define (problem BLOCKS-5-0)
(:domain blocksworld-fn-cond)
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
    (= (loc a) b)
    (= (loc b) c)
    (= (loc c) d)
    (= (loc d) e)
    (= (loc e) table)
))

)