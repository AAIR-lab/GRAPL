; <magic_json> {"min_blocks": 16, "max_blocks": 16, "blocks": 16, "bin_params": ["blocks"]}

(define (problem BW-rand-16)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15)
(:init (arm-empty) (clear b13) (on b13 b12) (on b12 b7) (on-table b7) (clear b2) (on b2 b1) (on b1 b3) (on b3 b5) (on b5 b14) (on b14 b9) (on b9 b4) (on b4 b8) (on b8 b11) (on-table b11) (clear b6) (on-table b6) (clear b10) (on b10 b0) (on b0 b15) (on-table b15))
(:goal (and (on b3 b2) (on b2 b11) (on b11 b6) (on b6 b5) (on b5 b14) (on b14 b8) (on b8 b9) (on b9 b12) (on b12 b13) (on b13 b1) (on b1 b10) (on b10 b7) (on b7 b4) (on b4 b0) (on b0 b15)))
)
