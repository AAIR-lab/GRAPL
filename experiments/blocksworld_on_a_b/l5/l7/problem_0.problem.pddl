; <magic_json> {"min_blocks": 16, "max_blocks": 16, "blocks": 16, "bin_params": ["blocks"]}

(define (problem BW-rand-16)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15)
(:init (arm-empty) (clear b15) (on b15 b6) (on b6 b12) (on b12 b4) (on b4 b8) (on b8 b11) (on-table b11) (clear b10) (on b10 b2) (on-table b2) (clear b3) (on b3 b7) (on b7 b1) (on b1 b5) (on b5 b14) (on-table b14) (clear b9) (on b9 b0) (on-table b0) (clear b13) (on-table b13))
(:goal (and (on b13 b9)))
)
