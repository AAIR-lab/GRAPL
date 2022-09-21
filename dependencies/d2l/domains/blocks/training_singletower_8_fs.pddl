(define (problem BLOCKS-10-0)
(:domain blocksworld-fn-cond)
(:objects a b c d e f g h - block)

  (:init
    (= (loc a) d)
	(= (loc d) table)

	(= (loc e) b)
	(= (loc b) c)
	(= (loc c) f)
	(= (loc f) table)

	(= (loc g) table)

	(= (loc h) table)

	(clear a)
	(clear e)
	(clear g)
	(clear h)
	(clear table)
  )

(:goal (and
    (= (loc a) b)
    (= (loc b) c)
    (= (loc c) d)
    (= (loc d) e)
    (= (loc e) f)
    (= (loc f) g)
    (= (loc g) h)
    (= (loc h) table)
))

)