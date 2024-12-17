(define (domain prob_domain) 
 (:requirements :strips :probabilistic-effects :conditional-effects) 
 (:constants YS SY SF NO EV SZ UQ WE VA VY )
 (:predicates 
	 (KB ?X ) 
	 (DJ ?X ?Y ) 
	 (WY ?X ) 
	 (QF ?X ) 
	 (DL ?X ?Y ) 
(clear)
(not-clear)
 )
(:action TNW
 :parameters (?X ?Y )
 :precondition (and 
		 (DL ?X ?Y) 
  )
 :effect (probabilistic 
		 100/100 (and (QF ?X) )  
          )
 )
(:action ORE
 :parameters (?X )
 :precondition (and 
		 (QF ?X) 
		 (WY ?X) 
  )
 :effect (probabilistic 
		 100/100 (and (not (QF ?X)) )  
          )
 )
(:action BED
 :parameters (?X )
 :precondition (and 
		 (DJ ?X ?X) 
  )
 :effect (probabilistic 
		 100/100 (and (not (DJ ?X ?X)) )  
          )
 )
(:action GKO
 :parameters (?X ?Y ?Z )
 :precondition (and 
		 (DL ?Z ?X) 
  )
 :effect (probabilistic 
		 89/100 (and (DL ?Z ?Y) (not (DL ?Z ?X)) )  
		 11/100 (and (KB ?X) )  
          )
 )
(:action NNN
 :parameters (?X ?Y ?Z )
 :precondition (and 
		 (WY ?Y) 
		 (DJ ?X ?Y) 
		 (DJ ?X ?Z) 
  )
 :effect (probabilistic 
		 100/100 (and (QF ?Z) (KB ?X) (QF ?Y) )  
          )
 )
(:action reset1 
 :precondition (not-clear)
 :effect (and 
	     (forall (?x) (and 
      (not (KB ?x)) 
      (not (DJ ?x YS)) 
      (not (DJ ?x SY)) 
      (not (DJ ?x SF)) 
      (not (DJ ?x NO)) 
      (not (DJ ?x EV)) 
      (not (DJ ?x SZ)) 
      (not (DJ ?x UQ)) 
      (not (DJ ?x WE)) 
      (not (DJ ?x VA)) 
      (not (DJ ?x VY)) 
      (not (WY ?x)) 
      (not (QF ?x)) 
      (not (DL ?x YS)) 
      (not (DL ?x SY)) 
      (not (DL ?x SF)) 
      (not (DL ?x NO)) 
      (not (DL ?x EV)) 
      (not (DL ?x SZ)) 
      (not (DL ?x UQ)) 
      (not (DL ?x WE)) 
      (not (DL ?x VA)) 
      (not (DL ?x VY)) 
))
(not (not-clear))
(clear)))

(:action reset2 
 :precondition (clear) 
 :effect (and (not-clear)
              (not (clear))
(DJ VA SY) 
(KB YS) 
(KB WE) 
(DL SY YS) 
(DL NO VA) 
(QF NO) 
(DJ UQ UQ) 
(WY NO) 
(KB NO) 
(DL UQ SZ) 
(DJ YS VA) 
(WY WE) 
(KB UQ) 
(QF UQ) 
(KB SY) 
(QF WE) 
(DJ EV SF) 
(DJ SZ SY) 
(QF SF) 
(DL SZ WE) 
)))
(define (problem random-problem538) 
 (:domain prob_domain) 
 (:init 
(not-clear)
(DJ VA SY) (KB YS) (KB WE) (DL SY YS) (DL NO VA) (QF NO) (DJ UQ UQ) (WY NO) (KB NO) (DL UQ SZ) (DJ YS VA) (WY WE) (KB UQ) (QF UQ) (KB SY) (QF WE) (DJ EV SF) (DJ SZ SY) (QF SF) (DL SZ WE)  
)
 (:goal (and 
(DL UQ  YS ) 
(DL SZ  NO ) 
(DL NO  UQ ) 
(DL SY  SY ) 
)))
