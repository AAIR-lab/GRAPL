; <magic_json> {"min_blocks": 8, "max_blocks": 8, "blocks": 8, "bin_params": ["blocks"]}

(define (problem BW-rand-8)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7)
(:init (arm-empty) (clear b5) (on b5 b7) (on b7 b0) (on b0 b3) (on b3 b1) (on-table b1) (clear b4) (on-table b4) (clear b2) (on-table b2) (clear b6) (on-table b6))
(:goal (and (on b4 b7)))
)
