#import "../lib.typ": *

#set outline(
  indent: 0pt
)

#set outline.entry(
  fill: none
)

#show outline: set heading(
  outlined: true,
)

#show outline: set align(left)

#show outline.entry: it => {
  v(12pt, weak: true)
  show "Table": none
  show "Figure": none
  if it.level == 1 and it.element.func() == heading {
    v(0.5em)
    strong(it)
  } else if it.element.func() == figure {
    link(it.element.location())[
      #grid(
        columns: (8em, 1fr, 6.5em),
        align: (left, left, right),
        it.prefix(),
        it.body(),
        it.page(),
      )
    ]
  } else if it.level == 3 {
    context {
      let num = counter(heading).at(it.element.location())
      link(it.element.location())[
        #grid(
          columns: (1fr, auto),
          align: (left, right),
          [
            #h(24pt)
            #numbering("1.1.1", ..num)
            #h(0.5em)
            #it.body()
          ],
          it.page(),
        )
      ]
    }
  } else if it.level >= 4 {
    hide(it)
  } else {
    it
  }
}

// --- TOC outline ---
#non_outlined_heading[TABLE OF CONTENTS]

#grid(
  columns: (1fr, auto),
  align: (left, center),
  stack(dir: ltr)[*Title*], [*Page No.*]
)

#outline(
  title: none,
  target: heading,
)

#pagebreak()

// Start roman page numbers
#set page(numbering: "I")
#counter(page).update(1)

// --- Tables list ---
#non_outlined_heading[LIST OF TABLES]

#grid(
  columns: (8em, 1fr, 6.5em),
  align: (left, left, right),
  [*Table No.*], [*Title*], [*Page No.*]
)
#outline(
  title: none,
  target: figure.where(kind: table),
)

#pagebreak()

// --- Figure list ---
#non_outlined_heading[LIST OF FIGURES]

#grid(
  columns: (8.5em, 1fr, 6.5em),
  align: (left, left, right),
  [*Figure No.*], [*Title*], [*Page No.*]
)
#outline(
  title: none,
  target: figure.where(kind: image),
)

#pagebreak()
