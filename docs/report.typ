#import "meta.typ": *
#import "lib.typ": *

#set document(
  title: document_title,
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
)

#set text(
  font: "Times New Roman",
  size: 12pt,
)

#set par(
    justify: true,
)

#set list(
  indent: 2em,
  spacing: 1.5em,
)

#set outline(
  indent: 0pt
)

#set outline.entry(
  fill: none
)

#show outline: set heading(
  outlined: true,
)

#show outline: set align(center)

#show heading: it => {
  v(0.5em)
  it
}

#show outline.entry: it => {
  v(12pt, weak: true)
  link(it.element.location())[
    #if it.level == 1 and it.element.func() == heading {
      v(0.5em)
      grid(
        columns: (1fr, auto),
        align: (left, center),
        stack(dir: ltr)[*#it.body()*], [*#it.page()*]
      )
    } else {
      grid(
        columns: (1fr, 10fr, auto),
        align: (left, left, right),
        stack(dir: ltr)[
          #{
            show "Table": none
            show "Figure": none
            it.prefix()
          }
        ], [#it.body()], [#it.page()]
      )
    }
  ]
}

#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)

#counter(heading).update(0)

// --- Front matter ---
#include "frontmatter/titlepage.typ"
#include "frontmatter/certificate.typ"
#include "frontmatter/declaration.typ"
#include "frontmatter/acknowledgement.typ"
#include "frontmatter/abstract.typ"
#include "frontmatter/toc.typ"

// --- Header and footer for the main body ---
#set page(
  paper: "a4",
  margin: (top: 3cm, bottom: 2.5cm, x: 2.5cm),
  numbering: "1",
  
  // Header definition
  header: [
      #set text(9pt)
      #header_title #h(1fr) #header_year
      #v(-0.8em)
      #header_footer_line()
  ],
  
  // Footer definition
  footer: context [
    #set text(9pt)
    #footer_line()
    #v(-0.8em)
    #footer_dept #h(1fr) Page #counter(page).display()
  ]
)

#counter(page).update(1)
#counter(heading).update(0)

#set heading(numbering: "1.1")

// --- Chapters ---
#include "chapters/chapter1.typ"
#include "chapters/chapter2.typ"
#include "chapters/chapter3.typ"
#include "chapters/chapter4.typ"
