(define (problem BLOCKS-10-0)
(:domain blocksworld-fn)
(:objects a b c d e f g h i j - block)

  (:init
    (= (loc a) d)
	(= (loc d) table)

	(= (loc e) b)
	(= (loc b) c)
	(= (loc c) f)
	(= (loc f) table)

	(= (loc g) table)

	(= (loc j) i)
	(= (loc i) h)
	(= (loc h) table)

	(clear a)
	(clear e)
	(clear g)
	(clear j)
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
    (= (loc h) i)
    (= (loc i) j)
    (= (loc j) table)
))

)