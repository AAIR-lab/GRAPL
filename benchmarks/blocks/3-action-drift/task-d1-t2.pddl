
(define (problem task-d1-t2) (:domain domain-d1)
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
	(holding b5)
	(on-table b1)
	(clear b6)
	(on b3 b4)
	(holding b6)
	(holding b3)
	(on-table b5)
	(on b2 b3)
	(clear b5)
	(on b4 b6)
	(holding b2)
	(not (on b5 b3))
	(not (clear b2))
	(not (on-table b2))
	(arm-empty)
	(on b6 b1)))
)