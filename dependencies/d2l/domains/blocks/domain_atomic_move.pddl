;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Atomic-move blocksworld
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (domain blocksworld-atomic)
  (:requirements :strips)

  (:constants table)

  (:predicates (on ?x ?y) (clear ?x) (diff ?x ?y))

  (:action move
    :parameters (?b  ?x ?y)
    :precondition (and (diff ?b table) (diff ?y table) (diff ?b ?y) (clear ?b) (on ?b ?x) (clear ?y))
    :effect (and (not (on ?b ?x)) (clear ?x) (not (clear ?y)) (on ?b ?y)))

  (:action move-to-table
    :parameters (?b ?x)
    :precondition (and (on ?b ?x) (clear ?b) (diff ?b table) (diff ?x table))
    :effect (and (not (on ?b ?x)) (clear ?x) (on ?b table)))
)

