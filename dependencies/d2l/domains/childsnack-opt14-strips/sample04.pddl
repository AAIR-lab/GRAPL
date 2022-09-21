(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child1 - child
    bread1 bread2 - bread-portion
    content1 content2 - content-portion
    tray1 tray2 - tray
    table1 table2 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray1 kitchen)
     (at tray2 kitchen)

     (at_kitchen_bread bread1)
     (at_kitchen_bread bread2)
     (at_kitchen_content content1)
     (at_kitchen_content content2)

     (no_gluten_bread bread1)
     (no_gluten_content content1)

     (allergic_gluten child1)

     (waiting child1 table1)

     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child1)
    )
  )
)