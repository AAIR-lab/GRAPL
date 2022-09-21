; <magic_json> {"min_blocks": 6, "max_blocks": 6, "blocks": 6, "bin_params": ["blocks"]}

(define (problem BW-rand-6)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5)
(:init (arm-empty) (clear b0) (on b0 b2) (on b2 b1) (on b1 b5) (on b5 b4) (on-table b4) (clear b3) (on-table b3))
(:goal (and (on b4 b5)))
)
