; <magic_json> {"min_blocks": 10, "max_blocks": 10, "blocks": 10, "bin_params": ["blocks"]}

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
(:init (arm-empty) (clear b9) (on b9 b3) (on b3 b0) (on b0 b5) (on-table b5) (clear b6) (on b6 b2) (on b2 b7) (on-table b7) (clear b8) (on-table b8) (clear b4) (on b4 b1) (on-table b1))
(:goal (and (on b4 b2)))
)
