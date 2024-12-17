
(define (problem task-d2-t1) (:domain domain-d2)
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
	(on-table b1)
	(not (holding b3))
	(not (on b6 b1))
	(on b6 b3)
	(on b3 b4)
	(on b1 b6)
	(clear b5)
	(holding b6)
	(holding b2)
	(on b4 b6)
	(clear b3)
	(on b3 b1)
	(not (clear b2))
	(on b5 b3)
	(on-table b2)
	(not (arm-empty))
	(clear b1)
	(on b2 b1)))
)
