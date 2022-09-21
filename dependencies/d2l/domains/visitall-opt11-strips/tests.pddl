(define (problem grid-3)
(:domain grid-visit-all)
(:objects 
	c0
	c1
	c2
	c3
- place
        
)
(:init
	(at-robot c1)
	(visited c1)
 	(connected c0 c1)
 	(connected c1 c0)
 	(connected c1 c2)
 	(connected c2 c1)
 	(connected c2 c3)
 	(connected c3 c2)
)
(:goal
(and 
	(visited c0)
	(visited c1)
	(visited c2)
	(visited c3)
)
)
)