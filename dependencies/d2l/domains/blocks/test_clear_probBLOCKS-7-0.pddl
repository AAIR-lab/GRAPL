(define (problem BLOCKS-7-0)
(:domain BLOCKS)
(:objects C F A B G D E )
(:INIT (CLEAR E) (ONTABLE D) (ON E G) (ON G B) (ON B A) (ON A F) (ON F C)
 (ON C D) (HANDEMPTY))
(:goal (AND (CLEAR A)))
)