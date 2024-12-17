(define (problem tireworld-5)
  (:domain tireworld)
  (:objects
  l-1-1 - location
  l-1-2 - location
  l-1-3 - location
  l-2-1 - location
  l-2-2 - location
  l-3-1 - location
  )
  (:init
  (vehicle-at l-3-1)
  (road l-1-1 l-1-2)
  (road l-1-2 l-1-3)
  (road l-1-1 l-2-1)
  (road l-1-2 l-2-2)
  (road l-2-1 l-1-2)
  (road l-2-2 l-1-3)
  (road l-2-1 l-3-1)
  (road l-3-1 l-2-2)
  (spare-in l-2-1)
  (spare-in l-2-2)
  (spare-in l-3-1)
  (not-flattire)

)
  (:goal (and (vehicle-at l-1-3))))