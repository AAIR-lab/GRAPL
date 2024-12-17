
(define (problem task-t1) (:domain domain-t1)
  (:objects
        c1 - coin
	e1 - elevator
	f1 - floor
	f2 - floor
	p1 - pos
	p2 - pos
  )
  (:init 
	(at f1 p1)
	(at f1 p2)
	(at f2 p1)
	(at f2 p2)
	(dec_f f1 f2)
	(dec_f f2 f1)
	(dec_p p1 p2)
	(dec_p p2 p1)
	(gate f1 p1)
	(gate f1 p2)
	(gate f2 p1)
	(gate f2 p2)
	(in e1 f1)
	(in e1 f2)
	(inside e1)
	(is-first-floor f1)
	(is-first-floor f2)
	(is-first-position p1)
	(is-first-position p2)
	(shaft e1 p1)
	(shaft e1 p2)
	(underground)
  )
  (:goal (and
	(is-first-floor f1)
	(is-first-floor f2)
	(not (underground))
	(not (inside e1))
	(dec_f f1 f2)
	(is-first-position p1)
	(in e1 f2)
	(shaft e1 p1)
	(not (at f1 p2))
	(dec_p p1 p2)
	(not (at f2 p1))
	(dec_p p2 p1)
	(gate f2 p2)
	(is-first-position p2)
	(dec_f f2 f1)
	(not (in e1 f1))
	(have c1)
	(gate f1 p2)
	(gate f1 p1)
	(not (at f1 p1))
	(not (at f2 p2))
	(shaft e1 p2)
	(gate f2 p1)))
)
