; <magic_json> {"min_blocks": 4, "max_blocks": 4, "blocks": 4, "bin_params": ["blocks"]}

(define (problem BW-rand-4)
(:domain blocksworld)
(:objects b0 b1 b2 b3)
(:init (arm-empty) (clear b3) (on-table b3) (clear b0) (on b0 b2) (on-table b2) (clear b1) (on-table b1))
(:goal (and (on b1 b0)))
)