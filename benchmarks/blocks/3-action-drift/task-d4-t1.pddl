
(define (problem task-d4-t1) (:domain domain-d4)
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
	(holding b2)
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
	(holding b4)
	(on b4 b2)
	(not (clear b5))
	(holding b1)
	(on b2 b5)
	(on b3 b4)
	(on b2 b4)
	(holding b2)
	(on b4 b6)
	(on b5 b2)
	(not (clear b2))
	(on b5 b3)
	(not (on-table b2))
	(not (arm-empty))
	(clear b1)
	(on b6 b1)))
)
