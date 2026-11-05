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
  hyphenate: false,
)

#show raw: set text(font: "Times New Roman", size: 12pt)

#set par(
  justify: true,
  //leading: 1.5em,
  leading: 1em,
  spacing: 1.5em,
  justification-limits: (spacing: (min: 100%, max: 100%)),
)

#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)

#counter(heading).update(0)

// --- Front matter ---
#include "frontmatter/titlepage.typ"
#include "frontmatter/certificate.typ"
#include "frontmatter/certificate_dairy.typ"

// Front-matter sheets below use the page geometry of the reference report.
// Declaration
#set page(
  margin: (top: 54pt, bottom: 14pt, left: 63.8pt, right: 54pt),
)
#include "frontmatter/declaration.typ"

// Acknowledgement
#set page(
  margin: (top: 66.1pt, bottom: 14pt, left: 63.8pt, right: 54pt),
)
#include "frontmatter/acknowledgement.typ"

// Abstract
#set page(
  margin: (top: 67.9pt, bottom: 14pt, left: 56.8pt, right: 39.1pt),
)
#include "frontmatter/abstract.typ"

// Restore the standard front-matter margins for the TOC, abbreviations, etc.
#set page(
  margin: (top: 0.75in, bottom: 0.75in, left: 1.25in, right: 1in),
)
#include "frontmatter/toc.typ"
#include "frontmatter/abbreviations.typ"

// --- Header and footer for the main body ---
#set page(
  paper: "a4",
  numbering: (cur, ..t) => if cur < 10 { "0" + str(cur) } else { str(cur) },
  
  // Header definition
  header: {
    set text(11pt)
    stack(
      dir: ttb,
      spacing: 5pt,
      [#header_title #h(1fr) #header_year],
      frame_double_rule(),
    )
  },

  // Footer definition
  footer: {
    set text(11pt)
    stack(
      dir: ttb,
      spacing: 5pt,
      frame_double_rule(),
      [#footer_dept #h(1fr) Page #context counter(page).display()],
    )
  },

  header-ascent: 5pt,
  footer-descent: 5pt,
)

#counter(page).update(1)
#counter(heading).update(0)

#set heading(numbering: "1.1.1")
#show heading: it => it
#show heading.where(level: 2): set text(size: 16pt, weight: "bold")
#show heading.where(level: 2): set block(above: 1.5em, below: 16pt)
#show heading.where(level: 3): set text(size: 14pt, weight: "bold")
#show heading.where(level: 3): set block(above: 1.5em, below: 15.5pt)

#set list(
  indent: 3em,
  spacing: 1.5em,
  marker: ([•], [•], [•], [•]),
)

// --- Chapters ---
#include "chapters/chapter1.typ"
#include "chapters/chapter2.typ"
#include "chapters/chapter3.typ"
#include "chapters/chapter4.typ"
#include "chapters/chapter5.typ"
#include "chapters/chapter6.typ"
#include "chapters/chapter7.typ"

// --- Bibliography ---
#include "bibliography.typ"

// --- Personal Profile ---
#include "personal_profile.typ"

// --- Pages after the personal profile: standalone, no report page formatting ---
#set page(
  header: none,
  footer: none,
  numbering: none,
)
#include "frontmatter/sdg.typ"

// --- Plagiarism result ---
#set page(margin: 0pt)
#align(center + horizon)[
  #image("/assets/Plaigarism-Result.pdf", width: 100%)
]

#pagebreak()

// JS certificate scans: bare full-page pages

#align(center + horizon)[
  #rotate(
    90deg,
    reflow: true,
    image("/assets/JS-Shreekrishna.png", width: 100%),
  )
]

#pagebreak()

#align(center + horizon)[
  #rotate(
    90deg,
    reflow: true,
    image("/assets/JS-chethan.jpeg", width: 100%),
  )
]

#pagebreak()

#align(center + horizon)[
  #rotate(
    90deg,
    reflow: true,
    image("/assets/JS-skanda.jpeg", width: 100%),
  )
]

#pagebreak()

#align(center + horizon)[
  #rotate(
    90deg,
    reflow: true,
    image("/assets/JS-Srinivas.jpeg", width: 100%),
  )
]

// --- Research paper (appended as-is, 11 pages) ---
#for p in range(1, 12) [
  #if p > 1 { pagebreak() }
  #image("/assets/Research-Paper.pdf", page: p, width: 100%)
]
