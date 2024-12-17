
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
	(clear b1)
	(clear b2)
	(clear b4)
	(clear b5)
	(holding b2)
	(holding b3)
	(on b3 b4)
	(on b4 b6)
	(on b5 b3)
	(on b6 b1)
	(on-table b1)
	(on-table b2)
  )
  (:goal (and
	(on-table b1)
	(on b3 b2)
	(on b3 b4)
	(on-table b5)
	(on-table b4)
	(on b1 b5)
	(not (arm-empty))
	(on b6 b1)
	(not (clear b4))
	(not (clear b2))
	(on-table b2)
	(on b5 b1)
	(not (clear b5))
	(not (holding b2))
	(on b4 b6)
	(on b5 b3)
	(on b4 b3)
	(clear b1)
	(not (holding b3))
	(clear b3)
	(on b2 b1)))
)
