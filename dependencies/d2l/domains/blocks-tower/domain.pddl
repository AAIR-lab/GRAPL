;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Blocksworld-tower domain
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; A variation of the classical blocksworld problem, functional version, where the objective
;;; is to stack all blocks in a single tower, regardless of block identities.
;;;
;;; This domain file is identical to the standard Blocksworld domain.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain fn-blocksworld-tower)

  (:types place - object
          block - place
  )

  (:constants table - place)

  (:predicates
   (clear ?b - place)
   )

  (:functions
   (loc ?b - block) - place
   )

  (:action move-to-block
   :parameters (?b - block ?from - place ?to - block)
   :precondition (and
              (clear ?b)
		      (clear ?to)
		      (= (loc ?b) ?from)
		      (not (= ?b ?to))
		      (not (= ?b ?from))
              (not (= ?from ?to)))
   :effect (and (assign (loc ?b) ?to)
		(clear ?from)
		(not (clear ?to))
		)
   )

  (:action move-to-table
   :parameters (?b ?from - block)
   :precondition (and (clear ?b)
                 (not (= ?b ?from))
                 (= (loc ?b) ?from))
   :effect (and (assign (loc ?b) table)
		(clear ?from))
   )

)