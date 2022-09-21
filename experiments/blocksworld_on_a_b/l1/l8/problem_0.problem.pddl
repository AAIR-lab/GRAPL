; <magic_json> {"min_blocks": 32, "max_blocks": 32, "blocks": 32, "bin_params": ["blocks"]}

(define (problem BW-rand-32)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31)
(:init (arm-empty) (clear b19) (on b19 b16) (on b16 b21) (on b21 b15) (on b15 b5) (on b5 b4) (on-table b4) (clear b23) (on b23 b8) (on b8 b10) (on b10 b27) (on b27 b9) (on b9 b26) (on b26 b28) (on b28 b13) (on b13 b31) (on b31 b6) (on b6 b11) (on b11 b1) (on b1 b3) (on-table b3) (clear b20) (on-table b20) (clear b2) (on b2 b18) (on b18 b22) (on b22 b0) (on b0 b24) (on b24 b17) (on b17 b25) (on b25 b7) (on b7 b30) (on b30 b29) (on-table b29) (clear b12) (on-table b12) (clear b14) (on-table b14))
(:goal (and (on b8 b0)))
)
