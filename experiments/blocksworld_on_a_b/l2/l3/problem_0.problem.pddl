; <magic_json> {"min_blocks": 6, "max_blocks": 6, "blocks": 6, "bin_params": ["blocks"]}

(define (problem BW-rand-6)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5)
(:init (arm-empty) (clear b4) (on b4 b5) (on-table b5) (clear b3) (on b3 b0) (on-table b0) (clear b1) (on b1 b2) (on-table b2))
(:goal (and (on b4 b0)))
)