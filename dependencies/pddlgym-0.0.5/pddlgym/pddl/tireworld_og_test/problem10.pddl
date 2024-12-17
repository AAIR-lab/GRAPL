(define (problem tireworld-10)
  (:domain tireworld)
  (:objects
  l-1-1 - location
  l-1-2 - location
  l-1-3 - location
  l-1-4 - location
  l-1-5 - location
  l-2-1 - location
  l-2-2 - location
  l-2-3 - location
  l-2-4 - location
  l-3-1 - location
  l-3-2 - location
  l-3-3 - location
  l-4-1 - location
  l-4-2 - location
  l-5-1 - location
  )
  (:init
  (vehicle-at l-3-1)
  (road l-1-1 l-1-2)
  (road l-1-2 l-1-3)
  (road l-1-3 l-1-4)
  (road l-1-4 l-1-5)
  (road l-1-1 l-2-1)
  (road l-1-2 l-2-2)
  (road l-1-3 l-2-3)
  (road l-1-4 l-2-4)
  (road l-2-1 l-1-2)
  (road l-2-2 l-1-3)
  (road l-2-3 l-1-4)
  (road l-2-4 l-1-5)
  (road l-3-1 l-3-2)
  (road l-3-2 l-3-3)
  (road l-2-1 l-3-1)
  (road l-2-3 l-3-3)
  (road l-3-1 l-2-2)
  (road l-3-3 l-2-4)
  (road l-3-1 l-4-1)
  (road l-3-2 l-4-2)
  (road l-4-1 l-3-2)
  (road l-4-2 l-3-3)
  (road l-4-1 l-5-1)
  (road l-5-1 l-4-2)
  (spare-in l-2-1)
  (spare-in l-2-2)
  (spare-in l-2-3)
  (spare-in l-2-4)
  (spare-in l-3-1)
  (spare-in l-3-3)
  (spare-in l-4-1)
  (spare-in l-4-2)
  (spare-in l-5-1)
  (not-flattire)
  )
  (:goal (and (vehicle-at l-1-5))))