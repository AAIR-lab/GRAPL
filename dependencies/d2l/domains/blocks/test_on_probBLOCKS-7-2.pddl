(define (problem BLOCKS-7-2)
(:domain BLOCKS)
(:objects E G C D F A B )
(:INIT (CLEAR B) (CLEAR A) (ONTABLE F) (ONTABLE D) (ON B C) (ON C G) (ON G E)
 (ON E F) (ON A D) (HANDEMPTY))
(:goal (AND (ON A B)))
)