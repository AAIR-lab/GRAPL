; <magic_json> {"min_blocks": 10, "max_blocks": 10, "blocks": 10, "bin_params": ["blocks"]}

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
(:init (arm-empty) (clear b4) (on b4 b7) (on b7 b9) (on-table b9) (clear b6) (on b6 b5) (on b5 b8) (on b8 b0) (on b0 b2) (on b2 b1) (on b1 b3) (on-table b3))
(:goal (and (on b2 b6)))
)
