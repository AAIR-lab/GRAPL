
(define (problem task-t1) (:domain domain-t1)
  (:objects
        obj0-o - blocks
	obj1-o - blocks
	obj2-o - blocks
	obj3-o - blocks
  )
  (:init 
	(arm-empty)
	(clear obj0-o)
	(clear obj1-o)
	(clear obj2-o)
	(clear obj3-o)
	(holding obj0-o)
	(holding obj1-o)
	(holding obj2-o)
	(holding obj3-o)
	(on-table obj0-o)
	(on-table obj1-o)
	(on-table obj2-o)
	(on-table obj3-o)
  )
  (:goal (and
	(on-table obj0-o)
	(clear obj3-o)
	(not (holding obj1-o))
	(holding obj0-o)
	(on-table obj2-o)
	(on obj1-o obj0-o)
	(clear obj1-o)
	(on obj2-o obj3-o)
	(not (on-table obj1-o))
	(on obj2-o obj1-o)
	(on-table obj3-o)
	(not (holding obj2-o))
	(not (clear obj0-o))
	(not (holding obj3-o))
	(clear obj2-o)
	(arm-empty)))
)
