
(define (problem task-t0) (:domain domain-t0)
  (:objects
        obj0-o - type0-t
	obj1-o - type0-t
	obj2-o - type0-t
	obj3-o - type0-t
	obj4-o - type0-t
	obj5-o - type0-t
  )
  (:init 
	(pred0-p obj1-o obj3-o)
	(pred0-p obj3-o obj0-o)
	(pred0-p obj4-o obj0-o)
	(pred1-p obj0-o obj3-o)
	(pred2-p obj3-o obj1-o)
	(pred3-p obj0-o obj1-o)
	(pred3-p obj3-o obj1-o)
	(pred4-p obj1-o obj0-o)
	(pred4-p obj3-o obj0-o)
  )
  (:goal (and
	(pred1-p obj1-o obj0-o)
	(not (pred3-p obj0-o obj1-o))
	(pred0-p obj4-o obj0-o)
	(pred0-p obj3-o obj0-o)
	(pred0-p obj0-o obj3-o)
	(pred4-p obj3-o obj0-o)
	(pred0-p obj1-o obj3-o)
	(not (pred3-p obj3-o obj1-o))
	(pred1-p obj0-o obj3-o)
	(pred2-p obj3-o obj1-o)
	(pred4-p obj1-o obj0-o)
	(pred1-p obj0-o obj1-o)))
)
