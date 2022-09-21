; <magic_json> {"min_blocks": 12, "max_blocks": 12, "blocks": 12, "bin_params": ["blocks"]}

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11)
(:init (arm-empty) (clear b4) (on b4 b3) (on b3 b7) (on b7 b8) (on b8 b0) (on b0 b1) (on-table b1) (clear b10) (on b10 b5) (on b5 b2) (on b2 b9) (on-table b9) (clear b11) (on-table b11) (clear b6) (on-table b6))
(:goal (and (on b2 b10) (on b10 b3) (on b3 b6) (on b6 b1) (on b1 b7) (on b7 b0) (on b0 b11) (on b11 b9) (on b9 b5) (on b5 b8) (on b8 b4)))
)
