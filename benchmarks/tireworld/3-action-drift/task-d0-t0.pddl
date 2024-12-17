
(define (problem task-d0-t0) (:domain domain-d0)
  (:objects
        l-1-1 - location
	l-1-2 - location
	l-1-3 - location
	l-2-1 - location
	l-2-2 - location
	l-3-1 - location
  )
  (:init 
	(not-flattire)
	(road l-1-1 l-1-2)
	(road l-1-1 l-2-1)
	(road l-1-2 l-1-3)
	(road l-1-2 l-2-2)
	(road l-2-1 l-1-2)
	(road l-2-1 l-3-1)
	(road l-2-2 l-1-3)
	(road l-3-1 l-2-2)
	(spare-in l-2-1)
	(spare-in l-2-2)
	(spare-in l-3-1)
	(vehicle-at l-2-1)
  )
  (:goal (and
	(vehicle-at l-1-3)))
)
