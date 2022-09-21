(define (problem grid-2)
(:domain grid-visit-all)
(:objects 
	loc-x0-y0
	loc-x0-y1
	loc-x1-y0
	loc-x1-y1
- place 
        
)
(:init
	(at-robot loc-x0-y1)
	(visited loc-x0-y1)
	(connected loc-x0-y0 loc-x1-y0)
 	(connected loc-x0-y0 loc-x0-y1)
 	(connected loc-x0-y1 loc-x1-y1)
 	(connected loc-x0-y1 loc-x0-y0)
 	(connected loc-x1-y0 loc-x0-y0)
 	(connected loc-x1-y0 loc-x1-y1)
 	(connected loc-x1-y1 loc-x0-y1)
 	(connected loc-x1-y1 loc-x1-y0)
 
)
(:goal
(and 
	(visited loc-x0-y0)
	(visited loc-x0-y1)
	(visited loc-x1-y0)
	(visited loc-x1-y1)
)
)
)
; <magic_json> {"min_size": 2, "max_size": 2, "min_g_percent": 100, "max_g_percent": 100, "min_hole_percent": 0, "max_hole_percent": 0, "size": 2, "goals": 4, "holes": 0, "seed": 501975345, "bin_params": ["size"]}

