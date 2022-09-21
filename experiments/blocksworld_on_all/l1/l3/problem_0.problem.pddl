; <magic_json> {"min_blocks": 6, "max_blocks": 6, "blocks": 6, "bin_params": ["blocks"]}

(define (problem BW-rand-6)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5)
(:init (arm-empty) (clear b5) (on b5 b2) (on b2 b0) (on b0 b1) (on b1 b3) (on b3 b4) (on-table b4))
(:goal (and (on b0 b2) (on b2 b4) (on b4 b1) (on b1 b5) (on b5 b3)))
)
