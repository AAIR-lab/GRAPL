
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
	(clear b1)
	(clear b2)
	(clear b5)
	(holding b3)
	(on b3 b4)
	(on b4 b6)
	(on b5 b3)
	(on b6 b1)
	(on-table b1)
	(on-table b2)
  )
  (:goal (and
	(clear b6)
	(not (clear b1))
	(not (holding b3))
	(not (clear b5))
	(on b6 b3)
	(holding b1)
	(on-table b3)
	(on b3 b4)
	(on b6 b5)
	(on b4 b6)
	(clear b3)
	(arm-empty)
	(on b5 b3)
	(on-table b2)
	(clear b2)
	(not (on-table b1))
	(on b6 b1)))
)
