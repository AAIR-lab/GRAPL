; <magic_json> {"min_blocks": 4, "max_blocks": 4, "blocks": 4, "bin_params": ["blocks"]}

(define (problem BW-rand-4)
(:domain blocksworld)
(:objects b0 b1 b2 b3)
(:init (arm-empty) (clear b1) (on b1 b3) (on-table b3) (clear b0) (on-table b0) (clear b2) (on-table b2))
(:goal (and (on b2 b1) (on b1 b0) (on b0 b3)))
)