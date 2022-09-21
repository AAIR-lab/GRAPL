; <magic_json> {"min_blocks": 4, "max_blocks": 4, "blocks": 4, "bin_params": ["blocks"]}

(define (problem BW-rand-4)
(:domain blocksworld)
(:objects b0 b1 b2 b3)
(:init (arm-empty) (clear b3) (on b3 b2) (on b2 b0) (on-table b0) (clear b1) (on-table b1))
(:goal (and (on b0 b1)))
)
