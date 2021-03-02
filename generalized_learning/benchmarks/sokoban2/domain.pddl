(define (domain sokoban)
(:requirements :typing)
(:types LOC BOX)
(:predicates 
             (at-robot ?l - LOC)
             (at ?o - BOX ?l - LOC)
             (adjacent_up ?l1 - LOC ?l2 - LOC)
             (adjacent_down ?l1 - LOC ?l2 - LOC)
             (adjacent_left ?l1 - LOC ?l2 - LOC)
             (adjacent_right ?l1 - LOC ?l2 - LOC)
             (clear ?l - LOC)
)


; up

(:action move_up
:parameters (?from - LOC ?to - LOC)
:precondition (and (clear ?to) (at-robot ?from) (adjacent_up ?from ?to))
:effect (and (at-robot ?to) (not (at-robot ?from))))
             

(:action push_up
:parameters  (?rloc - LOC ?bloc - LOC ?floc - LOC ?b - BOX)
:precondition (and (at-robot ?rloc) (at ?b ?bloc) (clear ?floc)
	           (adjacent_up ?rloc ?bloc) (adjacent_up ?bloc ?floc))

:effect (and (at-robot ?bloc) (at ?b ?floc) (clear ?bloc)
             (not (at-robot ?rloc)) (not (at ?b ?bloc)) (not (clear ?floc))))


; down

(:action move_down
:parameters (?from - LOC ?to - LOC)
:precondition (and (clear ?to) (at-robot ?from) (adjacent_down ?from ?to))
:effect (and (at-robot ?to) (not (at-robot ?from))))
             

(:action push_down
:parameters  (?rloc - LOC ?bloc - LOC ?floc - LOC ?b - BOX)
:precondition (and (at-robot ?rloc) (at ?b ?bloc) (clear ?floc)
	           (adjacent_down ?rloc ?bloc) (adjacent_down ?bloc ?floc))

:effect (and (at-robot ?bloc) (at ?b ?floc) (clear ?bloc)
             (not (at-robot ?rloc)) (not (at ?b ?bloc)) (not (clear ?floc))))

; left

(:action move_left
:parameters (?from - LOC ?to - LOC)
:precondition (and (clear ?to) (at-robot ?from) (adjacent_left ?from ?to))
:effect (and (at-robot ?to) (not (at-robot ?from))))
             

(:action push_left
:parameters  (?rloc - LOC ?bloc - LOC ?floc - LOC ?b - BOX)
:precondition (and (at-robot ?rloc) (at ?b ?bloc) (clear ?floc)
	           (adjacent_left ?rloc ?bloc) (adjacent_left ?bloc ?floc))

:effect (and (at-robot ?bloc) (at ?b ?floc) (clear ?bloc)
             (not (at-robot ?rloc)) (not (at ?b ?bloc)) (not (clear ?floc))))

; right

(:action move_right
:parameters (?from - LOC ?to - LOC)
:precondition (and (clear ?to) (at-robot ?from) (adjacent_right ?from ?to))
:effect (and (at-robot ?to) (not (at-robot ?from))))
             

(:action push_right
:parameters  (?rloc - LOC ?bloc - LOC ?floc - LOC ?b - BOX)
:precondition (and (at-robot ?rloc) (at ?b ?bloc) (clear ?floc)
	           (adjacent_right ?rloc ?bloc) (adjacent_right ?bloc ?floc))

:effect (and (at-robot ?bloc) (at ?b ?floc) (clear ?bloc)
             (not (at-robot ?rloc)) (not (at ?b ?bloc)) (not (clear ?floc)))))



