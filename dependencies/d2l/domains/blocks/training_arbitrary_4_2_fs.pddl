(define (problem BLOCKS-4-0)
(:domain blocksworld-fn-cond)
(:objects a b c d - block)

  (:init
    (= (loc a) table)
	(= (loc b) c)
	(= (loc c) d)
	(= (loc d) table)
	(clear a)
	(clear b)
	(clear table)
  )

(:goal (and
    (= (loc a) table)
    (= (loc b) table)
    (= (loc c) table)
    (= (loc d) table)
))

)