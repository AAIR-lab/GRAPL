
(define (problem task-d0-t1) (:domain domain-d0)
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
	(clear b6)
	(not (clear b5))
	(clear b4)
	(on-table b5)
	(holding b2)
	(not (on b5 b3))
	(clear b3)
	(on-table b4)
	(not (on b4 b6))
	(not (on-table b2))
	(on b3 b5)
	(not (arm-empty))
	(not (on b3 b4))
	(not (clear b2))
	(on b6 b1)))
)
