(define (problem p6)
 (:domain forest)
 (:objects
           x1 y1  x2 y2  x3 y3  x4 y4  x5 y5  x6 y6  - location
		 sx1 sy1  sx2 sy2  sx3 sy3  sx4 sy4  sx5 sy5  sx6 sy6  - sub-location
 )
 (:init 
	;;;;;;;;top level grid;;;;;;;
	(at-x x1) 
	(at-y y1)
(succ-loc x1 x2)
(succ-loc y1 y2)
(succ-loc x2 x3)
(succ-loc y2 y3)
(succ-loc x3 x4)
(succ-loc y3 y4)
(succ-loc x4 x5)
(succ-loc y4 y5)
(succ-loc x5 x6)
(succ-loc y5 y6)
	(ninitialized x1 y1)
	(ninitialized x1 y2)
	(ninitialized x1 y3)
	(ninitialized x1 y4)
	(ninitialized x1 y5)
	(ninitialized x1 y6)
	(ninitialized x2 y1)
	(ninitialized x2 y2)
	(ninitialized x2 y3)
	(ninitialized x2 y4)
	(ninitialized x2 y5)
	(ninitialized x2 y6)
	(ninitialized x3 y1)
	(ninitialized x3 y2)
	(ninitialized x3 y3)
	(ninitialized x3 y4)
	(ninitialized x3 y5)
	(ninitialized x3 y6)
	(ninitialized x4 y1)
	(ninitialized x4 y2)
	(ninitialized x4 y3)
	(ninitialized x4 y4)
	(ninitialized x4 y5)
	(ninitialized x4 y6)
	(ninitialized x5 y1)
	(ninitialized x5 y2)
	(ninitialized x5 y3)
	(ninitialized x5 y4)
	(ninitialized x5 y5)
	(ninitialized x5 y6)
	(ninitialized x6 y1)
	(ninitialized x6 y2)
	(ninitialized x6 y3)
	(ninitialized x6 y4)
	(ninitialized x6 y5)
	(ninitialized x6 y6)
	;;;;;;;;subproblems;;;;;;;;;;
  (problem-at x1 y1 blocksworld)
  (problem-at x1 y2 blocksworld)
  (problem-at x1 y3 blocksworld)
  (problem-at x1 y4 logistics)
  (problem-at x1 y5 logistics)
  (problem-at x1 y6 grid)
  (problem-at x2 y1 grid)
  (problem-at x2 y2 grid)
  (problem-at x2 y3 blocksworld)
  (problem-at x2 y4 blocksworld)
  (problem-at x2 y5 blocksworld)
  (problem-at x2 y6 logistics)
  (problem-at x3 y1 logistics)
  (problem-at x3 y2 grid)
  (problem-at x3 y3 grid)
  (problem-at x3 y4 grid)
  (problem-at x3 y5 blocksworld)
  (problem-at x3 y6 blocksworld)
  (problem-at x4 y1 logistics)
  (problem-at x4 y2 logistics)
  (problem-at x4 y3 blocksworld)
  (problem-at x4 y4 logistics)
  (problem-at x4 y5 logistics)
  (problem-at x4 y6 grid)
  (problem-at x5 y1 grid)
  (problem-at x5 y2 grid)
  (problem-at x5 y3 blocksworld)
  (problem-at x5 y4 blocksworld)
  (problem-at x5 y5 blocksworld)
  (problem-at x5 y6 logistics)
  (problem-at x6 y1 logistics)
  (problem-at x6 y2 grid)
  (problem-at x6 y3 grid)
  (problem-at x6 y4 grid)
  (problem-at x6 y5 blocksworld)
  (problem-at x6 y6 blocksworld)
        ;;;;;;;;enabling constraints;;
	(enabled x1 y1)
	(solved x1 y1)
  (enables x1 y1 x1 y2)
	(solved x1 y1)
  (enables x1 y2 x1 y3)
	(solved x1 y1)
  (enables x1 y3 x1 y4)
	(solved x1 y1)
  (enables x1 y4 x1 y5)
	(solved x1 y1)
  (enables x1 y5 x1 y6)
	(solved x1 y1)
	(solved x2 y2)
  (enables x2 y1 x2 y2)
	(solved x2 y2)
  (enables x1 y3 x2 y3)
	(solved x2 y2)
  (enables x1 y4 x2 y4)
	(solved x2 y2)
  (enables x2 y4 x2 y5)
	(solved x2 y2)
  (enables x1 y6 x2 y6)
	(solved x2 y2)
  (enables x2 y1 x3 y1)
	(solved x3 y3)
  (enables x3 y1 x3 y2)
	(solved x3 y3)
  (enables x3 y2 x3 y3)
	(solved x3 y3)
  (enables x2 y4 x3 y4)
	(solved x3 y3)
  (enables x3 y4 x3 y5)
	(solved x3 y3)
  (enables x3 y5 x3 y6)
	(solved x3 y3)
  (enables x3 y1 x4 y1)
	(solved x4 y4)
  (enables x3 y2 x4 y2)
	(solved x4 y4)
  (enables x4 y2 x4 y3)
	(solved x4 y4)
  (enables x3 y4 x4 y4)
	(solved x4 y4)
  (enables x3 y5 x4 y5)
	(solved x4 y4)
  (enables x4 y5 x4 y6)
	(solved x4 y4)
  (enables x4 y1 x5 y1)
	(solved x5 y5)
  (enables x4 y2 x5 y2)
	(solved x5 y5)
  (enables x5 y2 x5 y3)
	(solved x5 y5)
  (enables x5 y3 x5 y4)
	(solved x5 y5)
  (enables x4 y5 x5 y5)
	(solved x5 y5)
  (enables x5 y5 x5 y6)
	(solved x5 y5)
	(solved x6 y6)
  (enables x5 y2 x6 y2)
	(solved x6 y6)
  (enables x5 y3 x6 y3)
	(solved x6 y6)
  (enables x6 y3 x6 y4)
	(solved x6 y6)
  (enables x6 y4 x6 y5)
	(solved x6 y6)
  (enables x5 y6 x6 y6)
	(solved x6 y6)
	;;;;;;;;grid sub-problem;;;;;;
	(s-init-x sx6)
	(s-init-y sy6)
	(s-goal-x sx1)
	(s-goal-y sy1)
(s-succ-loc sx1 sx2)
(s-succ-loc sy1 sy2)
(s-succ-loc sx2 sx3)
(s-succ-loc sy2 sy3)
(s-succ-loc sx3 sx4)
(s-succ-loc sy3 sy4)
(s-succ-loc sx4 sx5)
(s-succ-loc sy4 sy5)
(s-succ-loc sx5 sx6)
(s-succ-loc sy5 sy6)
	;;;;;;logistics sub-problem;;
                (s-city-loc l11 c1) (s-city-loc l12 c1)
                (s-city-loc l21 c2) (s-city-loc l22 c2)
		(s-airport-loc l11) (s-airport-loc l21)
 )
 (:goal 
	(and (at-x x6) (at-y y6))
 )
)
