
(define (problem task-d3-t1) (:domain domain-d3)
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
	(clear b6)
	(holding b4)
	(on b3 b4)
	(on-table b5)
	(not (arm-empty))
	(on b4 b5)
	(on b6 b1)
	(not (clear b4))
	(on b5 b2)
	(not (on b4 b6))
	(not (clear b2))
	(on-table b2)
	(on-table b6)
	(not (clear b5))
	(not (holding b2))
	(on b5 b3)
	(clear b1)
	(not (holding b3))
	(on b6 b4)
	(clear b3)
	(on b3 b5)))
)
