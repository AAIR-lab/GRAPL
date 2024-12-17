(define (problem elevators1)
  (:domain elevators)
  (:objects 
    f1 - floor
    f2 - floor 
    p1 - pos
    p2 - pos 
    e1 - elevator
    c1 - coin)
    
  (:init
   
    (is-first-floor f1) 
    (is-first-position p1) 
    (underground) 
    
    (dec_f f2 f1)     
    (dec_p p2 p1) 

    (shaft e1 p1) 
    (shaft e1 p2) 

    (in e1 f1) 

    (coin-at c1 f2 p2) 

    (gate f2 p1)
    (gate f2 p2)
    )
  (:goal (and (at f1 p1)))
)
