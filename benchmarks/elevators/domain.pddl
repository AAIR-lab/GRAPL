(define (domain elevators)
  (:requirements :non-deterministic :negative-preconditions :equality :typing)
  (:types elevator floor pos coin)

  
  (:predicates 
    (dec_f ?f - floor ?g - floor) 
    (dec_p ?p - pos ?q - pos) 
    (in ?e - elevator ?f - floor) 
    (at ?f - floor ?p - pos) 
    (shaft ?e - elevator ?p - pos) 
    (inside ?e - elevator) 
    (gate ?f - floor ?p - pos) 
    (coin-at ?c - coin ?f - floor ?p - pos) 
    (have ?c - coin)
  
    (underground)
    (is-first-floor ?f - floor)
    (is-first-position ?p - pos)
  )
  
  (:action move-to-first-floor
    :parameters (?f1 - floor ?p1 - pos)
    :precondition (and (underground) (is-first-floor ?f1) (is-first-position ?p1) (not (at ?f1 ?p1)))
    :effect (and (not (underground)) (at ?f1 ?p1))
  )
    
  (:action go-up
    :parameters (?e - elevator ?f - floor ?nf - floor)
    :precondition (and (not (underground)) (dec_f ?nf ?f) (in ?e ?f))
    :effect (and (in ?e ?nf) (not (in ?e ?f)))
  )
  (:action go-down
    :parameters (?e - elevator ?f - floor ?nf - floor)
    :precondition (and (not (underground)) (dec_f ?f ?nf) (in ?e ?f))
    :effect (and (in ?e ?nf) (not (in ?e ?f)))
  )
  (:action step-in
    :parameters (?e - elevator ?f - floor ?p - pos)
    :precondition (and (not (underground)) (at ?f ?p) (in ?e ?f) (shaft ?e ?p))
    :effect (and (inside ?e) (not (at ?f ?p)))
  )
  (:action step-out
    :parameters (?e - elevator ?f - floor ?p - pos)
    :precondition (and (not (underground)) (inside ?e) (in ?e ?f) (shaft ?e ?p))
    :effect (and (at ?f ?p) (not (inside ?e)))
  )
  (:action move-left-gate
    :parameters (?f - floor ?p - pos ?np - pos)
    :precondition (and (not (underground)) (at ?f ?p) (dec_p ?p ?np) (gate ?f ?p))
    :effect (and (not (at ?f ?p)) (probabilistic  0.8 (and (at ?f ?np))
                                                  0.2 (and (underground))))
  )
  (:action move-left-nogate
    :parameters (?f - floor ?p - pos ?np - pos)
    :precondition (and (not (underground)) (at ?f ?p) (dec_p ?p ?np) (not (gate ?f ?p)))
    :effect (and (not (at ?f ?p)) (at ?f ?np))
  )
  (:action move-right-gate
    :parameters (?f - floor ?p - pos ?np - pos)
    :precondition (and (not (underground)) (at ?f ?p) (dec_p ?np ?p) (gate ?f ?p))
    :effect (and (not (at ?f ?p))  (probabilistic 0.8 (and (at ?f ?np))
                                                  0.2 (and (underground))))
  )
  (:action move-right-nogate
    :parameters (?f - floor ?p - pos ?np - pos)
    :precondition (and (not (underground)) (at ?f ?p) (dec_p ?np ?p) (not (gate ?f ?p)))
    :effect (and (not (at ?f ?p)) (at ?f ?np))
  )
  (:action collect
    :parameters (?c - coin ?f - floor ?p - pos)
    :precondition (and (not (underground)) (coin-at ?c ?f ?p) (at ?f ?p))
    :effect (and (have ?c) (not (coin-at ?c ?f ?p)))
  )
)
