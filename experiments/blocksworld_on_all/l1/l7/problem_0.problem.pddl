; <magic_json> {"min_blocks": 12, "max_blocks": 12, "blocks": 12, "bin_params": ["blocks"]}

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11)
(:init (arm-empty) (clear b3) (on b3 b0) (on b0 b6) (on b6 b7) (on-table b7) (clear b10) (on b10 b1) (on b1 b9) (on b9 b11) (on b11 b5) (on b5 b8) (on b8 b4) (on-table b4) (clear b2) (on-table b2))
(:goal (and (on b1 b11) (on b11 b5) (on b5 b3) (on b3 b4) (on b4 b8) (on b8 b6) (on b6 b10) (on b10 b7) (on b7 b9) (on b9 b2) (on b2 b0)))
)
