; <magic_json> {"min_blocks": 32, "max_blocks": 32, "blocks": 32, "bin_params": ["blocks"]}

(define (problem BW-rand-32)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31)
(:init (arm-empty) (clear b7) (on b7 b12) (on b12 b2) (on-table b2) (clear b30) (on b30 b25) (on b25 b14) (on b14 b19) (on b19 b15) (on b15 b20) (on b20 b26) (on b26 b9) (on b9 b5) (on b5 b17) (on b17 b28) (on b28 b4) (on b4 b23) (on b23 b11) (on b11 b0) (on-table b0) (clear b29) (on b29 b31) (on b31 b24) (on b24 b18) (on b18 b22) (on b22 b21) (on b21 b10) (on b10 b27) (on-table b27) (clear b13) (on b13 b16) (on-table b16) (clear b8) (on b8 b3) (on b3 b6) (on b6 b1) (on-table b1))
(:goal (and (on b23 b1)))
)
