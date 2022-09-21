; <magic_json> {"min_blocks": 16, "max_blocks": 16, "blocks": 16, "bin_params": ["blocks"]}

(define (problem BW-rand-16)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15)
(:init (arm-empty) (clear b9) (on b9 b10) (on b10 b14) (on b14 b5) (on b5 b1) (on-table b1) (clear b11) (on b11 b2) (on-table b2) (clear b4) (on b4 b8) (on-table b8) (clear b13) (on b13 b7) (on-table b7) (clear b6) (on-table b6) (clear b12) (on b12 b0) (on b0 b15) (on b15 b3) (on-table b3))
(:goal (and (on b12 b7) (on b7 b4) (on b4 b3) (on b3 b9) (on b9 b8) (on b8 b1) (on b1 b2) (on b2 b6) (on b6 b15) (on b15 b13) (on b13 b10) (on b10 b5) (on b5 b11) (on b11 b0) (on b0 b14)))
)
