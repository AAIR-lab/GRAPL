
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
	(on-table b6)
	(on-table b1)
	(not (holding b2))
	(on b3 b4)
	(holding b6)
	(holding b3)
	(on-table b5)
	(on b2 b3)
	(clear b5)
	(on b4 b6)
	(on b5 b2)
	(not (clear b2))
	(on b5 b3)
	(not (on-table b2))
	(arm-empty)
	(on b6 b1)))
)
