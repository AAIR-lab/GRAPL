; <magic_json> {"min_blocks": 7, "max_blocks": 7, "blocks": 7, "bin_params": ["blocks"]}

(define (problem BW-rand-7)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6)
(:init (arm-empty) (clear b3) (on b3 b0) (on b0 b6) (on b6 b5) (on b5 b4) (on-table b4) (clear b1) (on b1 b2) (on-table b2))
(:goal (and (on b4 b3) (on b3 b1) (on b1 b5) (on b5 b0) (on b0 b2) (on b2 b6)))
)
