; <magic_json> {"min_blocks": 8, "max_blocks": 8, "blocks": 8, "bin_params": ["blocks"]}

(define (problem BW-rand-8)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7)
(:init (arm-empty) (clear b7) (on b7 b4) (on-table b4) (clear b1) (on-table b1) (clear b2) (on b2 b3) (on-table b3) (clear b6) (on b6 b5) (on b5 b0) (on-table b0))
(:goal (and (on b0 b1)))
)
