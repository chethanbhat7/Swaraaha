#import "../lib.typ": *

#set outline(
  indent: 61pt
)

#set outline.entry(
  fill: none
)

#show outline: set heading(
  outlined: true,
)

#show outline: set align(center)

#show outline.entry: it => {
  v(12pt, weak: true)
  show "Table": none
  show "Figure": none
  if it.level == 1 and it.element.func() == heading {
    v(0.5em)
    strong(it)
  } else {
    it
  }
}

// --- TOC outline ---
#non_outlined_heading[TABLE OF CONTENT]

#grid(
  columns: (1fr, auto),
  align: (left, center),
  stack(dir: ltr)[*Title*], [*Page \ No.*]
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
#align(center)[= LIST OF TABLES]
#v(0.5em)

#grid(
  columns: (auto, 1fr, auto),
  align: center,
  stack(dir: ltr)[*Table \ No.*], [*Title*], [*Page \ No.*]
)
#outline(
  title: none,
  target: figure.where(kind: table),
)

#pagebreak()

// --- Figure list ---
#align(center)[= LIST OF FIGURES]
#v(0.5em)

#grid(
  columns: (auto, 1fr, auto),
  align: center,
  stack(dir: ltr)[*Table \ No.*], [*Title*], [*Page \ No.*]
)
#outline(
  title: none,
  target: figure.where(kind: image),
)

#pagebreak()
