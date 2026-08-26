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
    justification-limits: (spacing: (min: 100%, max: 150%)),
)

#set list(
  indent: 3em,
  spacing: 1.5em,
)

#show heading: it => {
  v(0.5em)
  it
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
#include "frontmatter/abbreviations.typ"

// --- Header and footer for the main body ---
#set page(
  paper: "a4",
  margin: (top: 53.85pt, bottom: 72pt, left: 87.9pt, right: 72pt),
  numbering: "1",
  
  // Header definition
  header: [
      #set text(11pt)
      #header_title #h(1fr) #header_year
      #v(-1.15em)
      #frame_double_rule()
  ],

  // Footer definition
  footer: {
    set text(11pt)
    frame_double_rule()
    v(-1.25em)
    footer_dept
    h(1fr)
    [Page ]
    context counter(page).display()
  }
)

#counter(page).update(1)
#counter(heading).update(0)

#set heading(numbering: "1.1.1")
#show heading: it => it
#show heading.where(level: 2): set text(size: 16pt, weight: "bold")
#show heading.where(level: 2): set block(above: 12.6pt, below: 16pt)
#show heading.where(level: 3): set text(size: 14pt, weight: "bold")
#show heading.where(level: 3): set block(above: 12.6pt, below: 10pt)

#set par(
  justify: true,
  leading: 12.37pt,
  spacing: 18.32pt,
  justification-limits: (spacing: (min: 100%, max: 150%)),
)

#set list(
  indent: 18pt,
  body-indent: 13pt,
  spacing: 18.32pt,
)

// --- Chapters ---
#include "chapters/chapter1.typ"
#include "chapters/chapter2.typ"
// #include "chapters/chapter3.typ"
// #include "chapters/chapter4.typ"
