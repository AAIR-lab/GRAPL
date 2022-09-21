; <magic_json> {"min_blocks": 12, "max_blocks": 12, "blocks": 12, "bin_params": ["blocks"]}

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11)
(:init (arm-empty) (clear b11) (on b11 b3) (on b3 b6) (on b6 b5) (on b5 b1) (on-table b1) (clear b4) (on b4 b0) (on b0 b8) (on b8 b2) (on b2 b7) (on b7 b10) (on b10 b9) (on-table b9))
(:goal (and (on b1 b6) (on b6 b11) (on b11 b0) (on b0 b5) (on b5 b10) (on b10 b2) (on b2 b4) (on b4 b3) (on b3 b8) (on b8 b7) (on b7 b9)))
)
