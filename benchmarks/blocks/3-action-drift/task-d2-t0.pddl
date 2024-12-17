
(define (problem task-d2-t0) (:domain domain-d2)
  (:objects
        b1 - blocks
	b2 - blocks
	b3 - blocks
	b4 - blocks
	b5 - blocks
	b6 - blocks
  )
  (:init 
	(arm-empty)
	(clear b2)
	(clear b5)
	(on b3 b4)
	(on b4 b6)
	(on b5 b3)
	(on b6 b1)
	(on-table b1)
	(on-table b2)
  )
  (:goal (and
	(on-table b1)
	(not (on b6 b1))
	(not (clear b5))
	(holding b1)
	(on b3 b4)
	(clear b4)
	(on b4 b1)
	(on b2 b3)
	(on b4 b6)
	(not (clear b2))
	(on b5 b3)
	(not (on-table b2))
	(arm-empty)))
)
