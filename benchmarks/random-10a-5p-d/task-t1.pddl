
(define (problem task-t1) (:domain domain-t1)
  (:objects
        obj0-o - type0-t
	obj1-o - type0-t
	obj2-o - type0-t
	obj3-o - type0-t
	obj4-o - type0-t
	obj5-o - type0-t
  )
  (:init 
	(pred0-p obj3-o obj2-o)
	(pred2-p obj3-o obj5-o)
	(pred3-p obj2-o obj3-o)
	(pred3-p obj3-o obj5-o)
	(pred4-p obj3-o obj2-o)
  )
  (:goal (and
	(pred2-p obj2-o obj3-o)
	(pred1-p obj3-o obj2-o)
	(not (pred0-p obj3-o obj2-o))
	(pred4-p obj3-o obj2-o)
	(pred2-p obj3-o obj5-o)
	(pred1-p obj2-o obj3-o)
	(not (pred3-p obj3-o obj5-o))
	(pred3-p obj2-o obj3-o)
	(pred2-p obj3-o obj2-o)
	(pred0-p obj2-o obj3-o)))
)
