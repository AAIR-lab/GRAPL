
(define (domain domain-d3)
(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)
(:types floor pos elevator coin)
(:predicates (dec_f ?v0 - floor ?v1 - floor) (dec_p ?v0 - pos ?v1 - pos) (in ?v0 - elevator ?v1 - floor) (at ?v0 - floor ?v1 - pos) (shaft ?v0 - elevator ?v1 - pos) (inside ?v0 - elevator) (gate ?v0 - floor ?v1 - pos) (coin-at ?v0 - coin ?v1 - floor ?v2 - pos) (have ?v0 - coin) (underground) (is-first-floor ?v0 - floor) (is-first-position ?v0 - pos))


	(:action move-to-first-floor
		:parameters (?f1 - floor ?p1 - pos)
		:precondition 
			(and (underground)
			(is-first-floor ?f1)
			(is-first-position ?p1)
			(not (at ?f1 ?p1)))
		:effect (and  (not (underground)) (at ?f1 ?p1) (probabilistic 1.000000 (and
			)))
	)


	(:action go-up
		:parameters (?e - elevator ?f - floor ?nf - floor)
		:precondition 
			(and (not (underground))
			(dec_f ?nf ?f)
			(in ?e ?f))
		:effect (and  (not (in ?e ?f)) (in ?e ?nf) (inside ?e) (probabilistic 1.000000 (and
			)))
	)


	(:action go-down
		:parameters (?e - elevator ?f - floor ?nf - floor)
		:precondition 
			(and (not (underground))
			(dec_f ?f ?nf)
			(in ?e ?f))
		:effect (and  (not (in ?e ?f)) (in ?e ?nf) (probabilistic 1.000000 (and
			)))
	)


	(:action step-in
		:parameters (?e - elevator ?f - floor ?p - pos)
		:precondition 
			(and (not (underground))
			(at ?f ?p)
			(in ?e ?f)
			(shaft ?e ?p))
		:effect (and  (not (at ?f ?p)) (inside ?e) (probabilistic 1.000000 (and
			)))
	)


	(:action step-out
		:parameters (?e - elevator ?f - floor ?p - pos)
		:precondition 
			(and (inside ?e)
			(not (in ?e ?f))
			(not (shaft ?e ?p)))
		:effect (and  (not (at ?f ?p)) (not (inside ?e)) (in ?e ?f) (probabilistic 1.000000 (and
			)))
	)


	(:action move-left-gate
		:parameters (?f - floor ?p - pos ?np - pos)
		:precondition 
			(and (not (underground))
			(at ?f ?p)
			(dec_p ?p ?np)
			(gate ?f ?p))
		:effect (and  (not (at ?f ?p)) (probabilistic 0.800000 (and
			(at ?f ?np)) 0.200000 (and
			(underground))))
	)


	(:action move-left-nogate
		:parameters (?f - floor ?p - pos ?np - pos)
		:precondition 
			(and (not (underground))
			(at ?f ?p)
			(dec_p ?p ?np)
			(not (gate ?f ?p)))
		:effect (and  (not (at ?f ?p)) (at ?f ?np) (probabilistic 1.000000 (and
			)))
	)


	(:action move-right-gate
		:parameters (?f - floor ?p - pos ?np - pos)
		:precondition 
			(and (at ?f ?p)
			(gate ?f ?p)
			(not (dec_p ?np ?p))
			(not (is-first-position ?np))
			(underground))
		:effect (and  (not (at ?f ?p)) (probabilistic 0.800000 (and
			(at ?f ?np)) 0.200000 (and
			(not (underground))
			(is-first-position ?np)
			(is-first-position ?np))))
	)


	(:action move-right-nogate
		:parameters (?f - floor ?p - pos ?np - pos)
		:precondition 
			(and (not (underground))
			(at ?f ?p)
			(dec_p ?np ?p)
			(not (gate ?f ?p)))
		:effect (and  (not (at ?f ?p)) (at ?f ?np) (probabilistic 1.000000 (and
			)))
	)


	(:action collect
		:parameters (?c - coin ?f - floor ?p - pos)
		:precondition 
			(and (not (underground))
			(coin-at ?c ?f ?p)
			(at ?f ?p))
		:effect (and  (not (coin-at ?c ?f ?p)) (have ?c) (probabilistic 1.000000 (and
			)))
	)
)