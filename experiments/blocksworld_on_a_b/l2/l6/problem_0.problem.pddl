; <magic_json> {"min_blocks": 10, "max_blocks": 10, "blocks": 10, "bin_params": ["blocks"]}

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
(:init (arm-empty) (clear b4) (on b4 b7) (on-table b7) (clear b5) (on b5 b6) (on-table b6) (clear b9) (on b9 b1) (on b1 b8) (on-table b8) (clear b0) (on-table b0) (clear b2) (on-table b2) (clear b3) (on-table b3))
(:goal (and (on b0 b5)))
)
