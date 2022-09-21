; <magic_json> {"min_blocks": 10, "max_blocks": 10, "blocks": 10, "bin_params": ["blocks"]}

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
(:init (arm-empty) (clear b1) (on b1 b7) (on b7 b0) (on b0 b8) (on b8 b9) (on b9 b6) (on b6 b2) (on b2 b4) (on-table b4) (clear b3) (on b3 b5) (on-table b5))
(:goal (and (on b0 b9) (on b9 b6) (on b6 b5) (on b5 b7) (on b7 b2) (on b2 b4) (on b4 b3) (on b3 b8) (on b8 b1)))
)
