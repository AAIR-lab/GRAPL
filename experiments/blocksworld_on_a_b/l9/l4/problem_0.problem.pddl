; <magic_json> {"min_blocks": 7, "max_blocks": 7, "blocks": 7, "bin_params": ["blocks"]}

(define (problem BW-rand-7)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6)
(:init (arm-empty) (clear b1) (on b1 b2) (on b2 b5) (on-table b5) (clear b0) (on b0 b4) (on-table b4) (clear b3) (on-table b3) (clear b6) (on-table b6))
(:goal (and (on b4 b2)))
)
