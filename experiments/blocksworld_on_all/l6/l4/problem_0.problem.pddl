; <magic_json> {"min_blocks": 7, "max_blocks": 7, "blocks": 7, "bin_params": ["blocks"]}

(define (problem BW-rand-7)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6)
(:init (arm-empty) (clear b6) (on-table b6) (clear b4) (on-table b4) (clear b2) (on-table b2) (clear b3) (on b3 b5) (on b5 b1) (on b1 b0) (on-table b0))
(:goal (and (on b3 b1) (on b1 b4) (on b4 b5) (on b5 b2) (on b2 b6) (on b6 b0)))
)
