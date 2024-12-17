(define (domain p2_knowledge)

(:requirements
    :negative-preconditions
    :typing
    :equality
)

(:types
location agent - object
)

(:predicates
    (at ?a - agent ?l - location)
    (trunk_empty ?a - agent)
    (luggage_location ?l - location)
    (reachable ?a - agent ?l - location)
    (knows_at ?other ?actor - agent ?l - location)
    (knows_trunk_empty ?other ?actor - agent)
    (knows_not_trunk_empty ?other ?actor - agent)
    (secret_content)
    (knows_reachable ?other ?actor - agent ?l - location)
    (decide)
    (destination ?actor - agent ?l - location)
)

(:action drive
    :parameters (?a - agent ?from ?to - location)
    :precondition (and
        (reachable ?a ?from)
        (reachable ?a ?to)
        (at ?a ?from)
    )
    :effect (and
        (at ?a ?to)
        (not (at ?a ?from))
    )
)


; I had to add extra constraints in load to ensure other agent is not in the same city
; else it uses transfers only
(:action unload_trunk
    :parameters (?actor ?other - agent ?l - location)
    :precondition (and
        (not (trunk_empty ?actor))
        (not (secret_content))
        (not (knows_at ?other ?actor ?l))
        (at ?actor ?l)
    )
    :effect (and
        (luggage_location ?l)
        (trunk_empty ?actor)
    )
)

(:action load_trunk
    :parameters (?actor - agent ?l - location)
    :precondition (and
        (trunk_empty ?actor)
        (luggage_location ?l)
        (at ?actor ?l)
    )
    :effect (and
        (not (luggage_location ?l))
        (not (trunk_empty ?actor))
        (secret_content)
    )
)

; privacy based version seems like a collaborative action
; even though acting agent is one
; Doesnt this action definition for each tells them what predicate do they need to know -- Why do we even need the graphplan ?
(:action transfer_trunk

    :parameters (?actor ?other - agent ?l - location)
    :precondition (and
        (at ?actor ?l)
         (not (= ?actor ?other))
        (at ?other ?l)
        (secret_content)
        (knows_at ?other ?actor ?l)
        (knows_trunk_empty ?other ?actor)     ; this should tell which predicate to communicate
        (not (trunk_empty ?actor))
    )
    :effect (and
        (trunk_empty ?actor)
        (not (trunk_empty ?other))
        (not (secret_content))
    )
)

(:action communicate_at
    :parameters (?actor ?other - agent ?l - location)
    :precondition (and
        (at ?other ?l))
    :effect (and
        (knows_at ?other ?actor ?l)
    )
)

(:action communicate_dest
    :parameters (?actor ?other - agent ?l - location)
    :precondition (and
        (decide))
    :effect (oneof (and (on ?b1 ?b2) (emptyhand) (clear ?b1) (not (holding ?b1)) (not (clear ?b2)))
                   (and (on-table ?b1) (emptyhand) (clear ?b1) (not (holding ?b1))))
)

(:action communicate_trunk_empty
    :parameters (?actor - agent ?other - agent ?l - location)
    :precondition (and
                    (at ?actor ?l)
                    (at ?other ?l)
                    (secret_content)
                    (not (= ?actor ?other))
                    )
    :effect (and
        (knows_at ?other ?actor ?l)
        (when
            (and (trunk_empty ?other))
            (and (knows_trunk_empty ?other ?actor))
        )
        (when
            (and (not (trunk_empty ?other)))
            (and (knows_not_trunk_empty ?other ?actor))
        )
    )
)

; (:action communicate_not_trunk_empty
;     :parameters (?actor - agent ?other - agent ?l - location)
;     :precondition (and
;                     (at ?actor ?l)
;                     (at ?other ?l)
;                     (not (= ?actor ?other))
;                     (not (trunk_empty ?actor))
;                     (secret_content))
;     :effect (and
;         (knows_not_trunk_empty ?other ?actor)

;     )
; )

; (:action communicate_trunk_empty
;     :parameters (?actor - agent ?other - agent ?l - location)
;     :precondition (and
;                     (at ?actor ?l)
;                     (at ?other ?l)
;                     (not (= ?actor ?other))
;                     (trunk_empty ?actor)
;                     (secret_content))
;     :effect (and
;         (knows_trunk_empty ?other ?actor)
;     )
; )

(:action communicate_reachable
    :parameters (?actor ?other - agent ?l - location)
    :precondition (and
        (reachable ?other ?l)
    )
    :effect (and
        (knows_reachable ?other ?actor ?l)
    )
)

)