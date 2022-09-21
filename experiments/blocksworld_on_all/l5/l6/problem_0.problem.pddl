; <magic_json> {"min_blocks": 10, "max_blocks": 10, "blocks": 10, "bin_params": ["blocks"]}

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
(:init (arm-empty) (clear b3) (on b3 b5) (on b5 b4) (on-table b4) (clear b8) (on b8 b1) (on b1 b7) (on b7 b6) (on b6 b9) (on b9 b0) (on b0 b2) (on-table b2))
(:goal (and (on b0 b3) (on b3 b7) (on b7 b1) (on b1 b2) (on b2 b6) (on b6 b5) (on b5 b4) (on b4 b9) (on b9 b8)))
)
