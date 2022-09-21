(define (problem BLOCKS-10-0)
(:domain blocksworld-fn-cond)
(:objects a b c d e f - block)

  (:init
    (= (loc a) d)
	(= (loc d) table)

	(= (loc e) b)
	(= (loc b) c)
	(= (loc c) f)
	(= (loc f) table)

	(clear a)
	(clear e)
	(clear table)
  )

(:goal (and
    (= (loc a) c)
    (= (loc b) table)
    (= (loc c) f)
    (= (loc d) e)
    (= (loc e) table)
    (= (loc f) table)
))

)