; <magic_json> {"min_blocks": 8, "max_blocks": 8, "blocks": 8, "bin_params": ["blocks"]}

(define (problem BW-rand-8)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7)
(:init (arm-empty) (clear b6) (on-table b6) (clear b5) (on b5 b1) (on b1 b3) (on-table b3) (clear b2) (on-table b2) (clear b4) (on-table b4) (clear b7) (on b7 b0) (on-table b0))
(:goal (and (on b3 b1)))
)
