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
    (= (loc a) b)
    (= (loc b) c)
    (= (loc c) d)
    (= (loc d) e)
    (= (loc e) f)
    (= (loc f) table)
))

)