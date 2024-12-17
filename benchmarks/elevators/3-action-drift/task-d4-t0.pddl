
(define (problem task-d4-t0) (:domain domain-d4)
  (:objects
        c1 - coin
	c2 - coin
	e1 - elevator
	f1 - floor
	f2 - floor
	p1 - pos
	p2 - pos
	p3 - pos
  )
  (:init 
	(coin-at c1 f1 p1)
	(coin-at c1 f1 p2)
	(coin-at c1 f2 p1)
	(coin-at c2 f2 p2)
	(dec_f f2 f1)
	(dec_p p1 p2)
	(dec_p p2 p1)
	(dec_p p2 p3)
	(dec_p p3 p2)
	(gate f1 p1)
	(gate f2 p1)
	(gate f2 p2)
	(in e1 f1)
	(is-first-floor f1)
	(is-first-position p1)
	(shaft e1 p1)
	(shaft e1 p2)
	(underground)
  )
  (:goal (and
	(is-first-floor f1)
	(not (underground))
	(have c2)
	(coin-at c1 f1 p1)
	(shaft e1 p3)
	(dec_p p3 p2)
	(dec_p p2 p3)
	(is-first-position p1)
	(in e1 f1)
	(shaft e1 p1)
	(dec_p p1 p2)
	(inside e1)
	(dec_p p2 p1)
	(gate f2 p2)
	(dec_f f2 f1)
	(have c1)
	(at f1 p2)
	(gate f1 p1)
	(at f1 p3)
	(shaft e1 p2)
	(gate f2 p1)))
)
