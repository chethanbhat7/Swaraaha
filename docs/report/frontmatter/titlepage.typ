#import "../lib.typ": *
#import "../meta.typ": *

// --- Title page ---
#[
  #set page(paper: "a4", margin: (top: 57.5pt, bottom: 54pt, left: 90pt, right: 72pt))
  #set text(font: "Times New Roman", hyphenate: false)
  #set par(leading: 0pt, spacing: 0pt)

  #let cl(size, body, weight: "regular", fill: black) = align(center, text(size: size, weight: weight, fill: fill)[#body])

  // -- university header --
  #cl(16pt, weight: "bold", fill: rgb("#c00000"))[#university_title]
  #v(7.4pt)
  #cl(12pt, weight: "bold")[#university_address]

  // -- vtu logo --
  #v(4.05pt)
  #align(center, move(dx: -5.8pt, image("../assets/vtu_logo.png", width: 101.2pt, height: 130.4pt)))

  // -- report type / on / project title --
  #v(8.55pt)
  #cl(12pt, weight: "bold")[#report_type]
  #v(9.0pt)
  #cl(12pt, weight: "bold")[on]
  #v(12.2pt)
  #cl(16pt, weight: "bold", fill: rgb("#ff0000"))[“#project_title”]
  #v(17.8pt)
  #cl(14pt, weight: "bold")[Submitted in partial fulfilment of the requirements for the award of]
  #v(11.5pt)
  #cl(16pt, weight: "bold")[#degree_upper]
  #v(12.2pt)
  #cl(12pt, weight: "bold")[in]
  #v(10.6pt)
  #cl(14pt, weight: "bold", fill: rgb("#7030a0"))[#department_upper]

  // -- submitted by --
  #v(21.7pt)
  #cl(13pt, weight: "bold")[Submitted By]
  #v(6.15pt)
  #grid(columns: (30pt, 1fr, 70pt, 95pt), column-gutter: 0pt,
    [],
    align(left, text(size: 13pt, weight: "bold", fill: rgb("#002060"))[Name]),
    align(right, text(size: 13pt, weight: "bold", fill: rgb("#002060"))[USN]),
    [],
  )
  #v(6.05pt)
  #for a in authors [
    #grid(columns: (30pt, 1fr, 70pt, 95pt), column-gutter: 0pt,
      [],
      align(left, text(size: 13pt, weight: "bold", fill: rgb("#0070c0"))[#upper(a.name)]),
      align(right, text(size: 13pt, weight: "bold", fill: rgb("#0070c0"))[#a.usn]),
      [],
    )
    #v(6.0pt)
  ]

  // -- guidance --
  #v(11.0pt)
  #cl(13pt, weight: "bold")[Under the Guidance of]
  #v(6.2pt)
  #cl(13pt, weight: "bold", fill: rgb("#c00000"))[#guide]
  #v(5.4pt)
  #cl(12pt, weight: "bold")[#guide_designation]

  // -- vcet logo --
  #v(30.6pt)
  #align(center, image("../assets/vcet_logo.png", width: 165pt, height: 102.1pt))
  #v(9.8pt)
  #place(top + left, dy: 597pt, line(length: 100%, stroke: 1.3pt))

  // -- college block --
  #cl(12pt, weight: "bold", fill: rgb("#0070c0"))[DEPARTMENT OF #department_upper]
  #v(7.7pt)
  #cl(14pt, weight: "bold", fill: rgb("#c00000"))[#college_upper]
  #v(8.9pt)
  #cl(10pt)[#college_unit]
  #v(6.2pt)
  #cl(9pt, fill: rgb("#0000ff"))[#affiliation_line1 #affiliation_line2]
  #v(5.9pt)
  #cl(12pt)[#college_address_cover]
  #v(7.5pt)
  #cl(12pt, weight: "bold", fill: rgb("#ff0000"))[#submission_month]
]
