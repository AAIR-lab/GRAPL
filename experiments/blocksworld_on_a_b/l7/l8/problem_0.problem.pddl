; <magic_json> {"min_blocks": 32, "max_blocks": 32, "blocks": 32, "bin_params": ["blocks"]}

(define (problem BW-rand-32)
(:domain blocksworld)
(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31)
(:init (arm-empty) (clear b11) (on b11 b28) (on b28 b3) (on b3 b23) (on b23 b26) (on b26 b29) (on b29 b1) (on b1 b9) (on b9 b7) (on b7 b21) (on-table b21) (clear b18) (on-table b18) (clear b14) (on-table b14) (clear b31) (on b31 b12) (on b12 b13) (on b13 b24) (on b24 b6) (on b6 b25) (on b25 b19) (on b19 b20) (on b20 b8) (on b8 b0) (on b0 b17) (on b17 b27) (on b27 b10) (on b10 b15) (on b15 b5) (on-table b5) (clear b22) (on-table b22) (clear b30) (on b30 b2) (on b2 b4) (on b4 b16) (on-table b16))
(:goal (and (on b23 b1)))
)
