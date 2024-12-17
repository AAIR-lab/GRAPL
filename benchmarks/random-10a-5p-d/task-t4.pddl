
(define (problem task-t4) (:domain domain-t4)
  (:objects
        obj0-o - type0-t
	obj1-o - type0-t
	obj2-o - type0-t
	obj3-o - type0-t
	obj4-o - type0-t
	obj5-o - type0-t
  )
  (:init 
	(pred1-p obj2-o obj0-o)
	(pred4-p obj0-o obj2-o)
	(pred4-p obj2-o obj0-o)
  )
  (:goal (and
	(pred1-p obj2-o obj0-o)
	(pred4-p obj0-o obj2-o)
	(pred4-p obj2-o obj0-o)
	(pred0-p obj2-o obj0-o)
	(pred3-p obj2-o obj0-o)
	(pred2-p obj2-o obj0-o)
	(pred2-p obj0-o obj2-o)
	(pred3-p obj0-o obj2-o)))
)
