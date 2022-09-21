; <magic_json> {"min_blocks": 16, "max_blocks": 16, "blocks": 16, "bin_params": ["blocks"]}

(define (problem BW-rand-16)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15)
(:init (arm-empty) (clear b13) (on b13 b2) (on b2 b4) (on b4 b9) (on b9 b0) (on b0 b11) (on b11 b10) (on b10 b12) (on b12 b14) (on b14 b6) (on b6 b7) (on b7 b15) (on-table b15) (clear b5) (on b5 b1) (on b1 b3) (on-table b3) (clear b8) (on-table b8))
(:goal (and (on b11 b15) (on b15 b13) (on b13 b0) (on b0 b8) (on b8 b9) (on b9 b6) (on b6 b4) (on b4 b14) (on b14 b2) (on b2 b10) (on b10 b1) (on b1 b5) (on b5 b12) (on b12 b3) (on b3 b7)))
)
