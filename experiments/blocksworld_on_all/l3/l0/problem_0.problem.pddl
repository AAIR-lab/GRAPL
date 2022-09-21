; <magic_json> {"min_blocks": 3, "max_blocks": 3, "blocks": 3, "bin_params": ["blocks"]}

(define (problem BW-rand-3)
(:domain blocksworld)
(:objects b0 b1 b2)
(:init (arm-empty) (clear b2) (on-table b2) (clear b0) (on-table b0) (clear b1) (on-table b1))
(:goal (and (on b2 b0) (on b0 b1)))
)
