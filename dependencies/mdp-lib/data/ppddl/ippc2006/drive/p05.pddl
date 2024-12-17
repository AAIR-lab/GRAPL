(define (domain drive)
        (:requirements :typing :probabilistic-effects :conditional-effects :disjunctive-preconditions :equality)
 
        (:types coord direction color delay preference length rotation)
 
        (:predicates
                (heading ?d - direction)
                (clockwise ?d1 ?d2 - direction)
                (at ?x - coord ?y - coord)
                (nextx ?a - coord ?b - coord ?h - direction)
                (nexty ?a - coord ?b - coord ?h - direction)
                (light_color ?c - color)
                (light_delay ?x ?y - coord ?d - delay)
                (light_preference ?x ?y - coord ?p - preference)
                (road-length ?start-x ?start-y ?end-x ?end-y - coord ?l - length)
                (alive)
        )
 
        (:constants 
                left right straight - rotation 
                north south east west - direction
                green red unknown - color
                quick normal slow - delay
                north_south none east_west - preference 
                short medium long - length
        )
 
        (:action look_at_light
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                        )
                :effect (and
                           (probabilistic
                             9/10
                             (when (and (heading north) (light_preference ?x ?y north_south))
                                   (and (not (light_color unknown))(light_color green)))
                             1/10 
                             (when (and (heading north) (light_preference ?x ?y north_south))
                                    (and (not (light_color unknown))(light_color red))))
                           (probabilistic
                             9/10
                             (when (and (heading south) (light_preference ?x ?y north_south))
                                   (and (not (light_color unknown))(light_color green)))
                             1/10 
                             (when (and (heading south) (light_preference ?x ?y north_south))
                                   (and (not (light_color unknown))(light_color red))))
                           (probabilistic
                             1/10
                             (when (and (heading east) (light_preference ?x ?y north_south))
                                   (and (not (light_color unknown))(light_color green)))
                             9/10
                             (when (and (heading east) (light_preference ?x ?y north_south))
                                   (and (not (light_color unknown))(light_color red))))

                           (probabilistic
                             1/10
                             (when (and (heading west) (light_preference ?x ?y north_south))
                                    (and (not (light_color unknown))(light_color green)))
                             9/10
                             (when (and (heading west) (light_preference ?x ?y north_south))
                                   (and (not (light_color unknown))(light_color red))))

                           (probabilistic
                             1/10
                             (when (and (heading north) (light_preference ?x ?y east_west))
                                   (and (not (light_color unknown))(light_color green)))
                             9/10
                             (when (and (heading north) (light_preference ?x ?y east_west))
                                   (and (not (light_color unknown))(light_color red))))
                           (probabilistic
                             1/10
                            (when (and (heading south) (light_preference ?x ?y east_west))
                                  (and (not (light_color unknown))(light_color green)))
                             9/10
                            (when (and (heading south) (light_preference ?x ?y east_west))
                                  (and (not (light_color unknown))(light_color red))))
                           (probabilistic
                             9/10
                             (when (and (heading east) (light_preference ?x ?y east_west))
                                   (and (not (light_color unknown))(light_color green)))
                             1/10
                             (when (and (heading east) (light_preference ?x ?y east_west))
                                   (and (not (light_color unknown))(light_color red))))

                           (probabilistic
                             9/10
                             (when (and (heading west) (light_preference ?x ?y east_west))
                                   (and (not (light_color unknown))(light_color green)))
                             1/10
                             (when (and (heading west) (light_preference ?x ?y east_west))
                                   (and (not (light_color unknown))(light_color red))))
                            (probabilistic
                             1/2
                             (when (light_preference ?x ?y none)
                                   (and (not (light_color unknown))(light_color green)))
                             1/2
                             (when (light_preference ?x ?y none)
                                   (and (not (light_color unknown))(light_color red)))))

        )

        (:action wait_on_light
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color red)
                                (at ?x ?y)
                        )
                :effect (and
                            (probabilistic
                                1/100 (not (alive))
                            )
                            (probabilistic
                              1/2
                              (when (light_delay ?x ?y quick)
                                    (and (not (light_color red))(light_color green))))

                            (probabilistic
                              1/5
                              (when (light_delay ?x ?y normal)
                                    (and (not (light_color red))(light_color green))))

                            (probabilistic
                              1/10
                              (when (light_delay ?x ?y slow)
                                    (and (not (light_color red))(light_color green))))
                        )
        )

        (:action proceed
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?old-heading ?new-heading - direction
                             ?length - length ?turn-direction - rotation)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?old-heading)
                                (or (and (= ?turn-direction right)
                                         (clockwise ?old-heading ?new-heading))
                                    (and (= ?turn-direction left)
                                         (clockwise ?new-heading ?old-heading))
                                    (and (= ?turn-direction straight)
                                         (= ?new-heading ?old-heading)))
                                (nextx ?x ?new-x ?new-heading)
                                (nexty ?y ?new-y ?new-heading)
                                (road-length ?x ?y ?new-x ?new-y ?length))
                :effect (and
                                (not (light_color green))
                                (light_color unknown)
                                (probabilistic
                                  1/50 
                                  (when (= ?length short)
                                        (not (alive))))
                                (probabilistic
                                  1/20 
                                  (when (= ?length medium)
                                        (not (alive))))
                                (probabilistic
                                  1/10 
                                  (when (= ?length long)
                                        (not (alive))))
                                (not (at ?x ?y))
                                (not (heading ?old-heading))
                                (heading ?new-heading)
                                (at ?new-x ?new-y))
        )
)
(define (problem a-drive-problem70)
(:domain drive)
(:objects c0 c1 c2  - coord)
(:init 
       (heading east)
       (at c0 c0)
       (alive)
       (light_color unknown)
       (clockwise north east)
       (clockwise east south)
       (clockwise south west)
       (clockwise west north)
       (nextx c0 c1 east)
       (nextx c1 c0 west)
       (nextx c1 c2 east)
       (nextx c2 c1 west)
       (nextx c0 c0 north)
       (nextx c0 c0 south)
       (nextx c1 c1 north)
       (nextx c1 c1 south)
       (nextx c2 c2 north)
       (nextx c2 c2 south)
       (nexty c0 c1 north)
       (nexty c1 c0 south)
       (nexty c1 c2 north)
       (nexty c2 c1 south)
       (nexty c0 c0 east)
       (nexty c0 c0 west)
       (nexty c1 c1 east)
       (nexty c1 c1 west)
       (nexty c2 c2 east)
       (nexty c2 c2 west)
       (light_delay c0 c0 quick)
       (light_delay c0 c1 quick)
       (light_delay c0 c2 quick)
       (light_delay c1 c0 quick)
       (light_delay c1 c1 normal)
       (light_delay c1 c2 normal)
       (light_delay c2 c0 quick)
       (light_delay c2 c1 quick)
       (light_delay c2 c2 slow)
       (light_preference c0 c0 east_west)
       (light_preference c0 c1 north_south)
       (light_preference c0 c2 east_west)
       (light_preference c1 c0 east_west)
       (light_preference c1 c1 east_west)
       (light_preference c1 c2 none)
       (light_preference c2 c0 north_south)
       (light_preference c2 c1 none)
       (light_preference c2 c2 east_west)
       (road-length c0 c0 c0 c1 short)
       (road-length c0 c1 c0 c0 short)
       (road-length c0 c1 c0 c2 long)
       (road-length c0 c2 c0 c1 long)
       (road-length c1 c0 c1 c1 medium)
       (road-length c1 c1 c1 c0 medium)
       (road-length c1 c1 c1 c2 medium)
       (road-length c1 c2 c1 c1 medium)
       (road-length c2 c0 c2 c1 short)
       (road-length c2 c1 c2 c0 short)
       (road-length c2 c1 c2 c2 long)
       (road-length c2 c2 c2 c1 long)
       (road-length c0 c0 c1 c0 short)
       (road-length c1 c0 c0 c0 short)
       (road-length c1 c0 c2 c0 short)
       (road-length c2 c0 c1 c0 short)
       (road-length c0 c1 c1 c1 medium)
       (road-length c1 c1 c0 c1 medium)
       (road-length c1 c1 c2 c1 short)
       (road-length c2 c1 c1 c1 short)
       (road-length c0 c2 c1 c2 short)
       (road-length c1 c2 c0 c2 short)
       (road-length c1 c2 c2 c2 short)
       (road-length c2 c2 c1 c2 short)
)
(:goal (and (alive) (at c2 c2))))
