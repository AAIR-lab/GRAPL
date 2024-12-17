
(define (problem None) (:domain domain-t0)
  (:objects
        obj0-o - type0-t
	obj1-o - type0-t
	obj2-o - type0-t
	obj3-o - type0-t
  )
  (:init 
	(pred0-p obj0-o obj2-o)
	(pred2-p obj2-o obj0-o)
  )
  (:goal (and
	(pred0-p obj1-o obj3-o)
	(pred0-p obj2-o obj1-o)
	(pred1-p obj0-o)
	(pred0-p obj1-o obj0-o)
	(pred0-p obj0-o obj3-o)
	(pred0-p obj0-o obj2-o)))
)
