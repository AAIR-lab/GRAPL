(define (problem strips-gripper-x-1)
   (:domain gripper-strips)
   (:objects rooma roomb roomc ball3 ball2 ball1 left1 right1 rob1 left2 right2 rob2)
   (:init (room rooma)
          (room roomb)
          (room roomc)

          (ball ball3)
          (ball ball2)
          (ball ball1)

          (robot rob1)
          (robot rob2)

          (at-robby rob1 rooma)
          (at-robby rob2 roomb)

          (free left1)
          (free right1)
          (free left2)
          (free right2)

          (at ball3 roomc)
          (at ball2 rooma)
          (at ball1 rooma)

          (gripper left1 rob1)
          (gripper right1 rob1)
          (gripper left2 rob2)
          (gripper right2 rob2)

          )

   (:goal (and (at ball3 roomb)
               (at ball2 roomb)
               (at ball1 roomb))))