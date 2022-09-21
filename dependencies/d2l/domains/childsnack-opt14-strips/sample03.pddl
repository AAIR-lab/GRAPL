(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child1 child2 child3 child4 child5 child6 child7 child8 - child
    bread1 - bread-portion
    content1 - content-portion
    tray1 tray2 - tray
    table1 table2 table3 - place
    sandw1 - sandwich
  )
  (:init
     (at tray1 kitchen)
     (at tray2 kitchen)

     (at_kitchen_bread bread1)
     (at_kitchen_content content1)

     (no_gluten_bread bread1)
     (no_gluten_content content1)

     (not_allergic_gluten child1)
     (not_allergic_gluten child2)
     (allergic_gluten child3)
     (allergic_gluten child4)
     (allergic_gluten child5)
     (allergic_gluten child6)
     (allergic_gluten child7)
     (allergic_gluten child8)

     (waiting child1 table1)
     (waiting child2 table2)
     (waiting child3 table2)
     (waiting child4 table2)
     (waiting child5 table2)
     (waiting child6 table2)
     (waiting child7 table2)
     (waiting child8 table2)

     (notexist sandw1)
  )
  (:goal
    (and
     (served child1)
     (served child2)
     (served child3)
     (served child4)
     (served child5)
     (served child6)
     (served child7)
     (served child8)
    )
  )
)