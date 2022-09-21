(define (problem strips-log-sample-1)
   (:domain logistics-strips)
   (:objects package1
             city2 city1
             truck1 plane1
             city1-1 city1-2 city2-1)
   (:init
          (obj package1)

          (city city2)
          (city city1)

          (truck truck1)
          (airplane plane1)

          (location city1-1)
          (location city1-2)
          (location city2-1)

          (airport city1-2)
          (airport city2-1)

          (in-city city1-1 city1)
          (in-city city1-2 city1)
          (in-city city2-1 city2)

          (at plane1 city1-2)
          (at truck1 city1-1)
          (at package1 city2-1))
   (:goal (and
               (at package1 city1-1))))