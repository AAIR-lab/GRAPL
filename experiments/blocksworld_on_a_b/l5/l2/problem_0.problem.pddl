; <magic_json> {"min_blocks": 5, "max_blocks": 5, "blocks": 5, "bin_params": ["blocks"]}

(define (problem BW-rand-5)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4)
(:init (arm-empty) (clear b1) (on b1 b2) (on b2 b3) (on b3 b0) (on-table b0) (clear b4) (on-table b4))
(:goal (and (on b1 b4)))
)
