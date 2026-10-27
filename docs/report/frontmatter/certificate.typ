#import "../lib.typ": *
#import "../meta.typ": *

// --- Certificate ---
#[
  #set page(paper: "a4", margin: (top: 54pt, bottom: 54pt, left: 54pt, right: 54pt))
  #set text(font: "Times New Roman")
  #set par(leading: 0pt, spacing: 0pt)

  #let cl(size, body, weight: "regular", fill: black) = align(center, text(size: size, weight: weight, fill: fill)[#body])
  #let purple = rgb("#7030a0")

  // -- college header --
  #cl(14pt, weight: "bold", fill: rgb("#c00000"))[#college_upper]
  #v(7.3pt)
  #cl(10pt)[#college_unit]
  #v(6.4pt)
  #cl(9pt, fill: rgb("#0000ff"))[#affiliation]
  #v(6.4pt)
  #cl(12pt)[#college_address]

  // -- department + logo + certificate --
  #v(23.9pt)
  #cl(13pt, weight: "bold", fill: rgb("#0070c0"))[DEPARTMENT OF #department_upper]
  #v(10.1pt)
  #align(center, image("../assets/vcet_logo.png", width: 165pt, height: 102.1pt))
  #v(10.8pt)
  #cl(18pt, weight: "bold")[CERTIFICATE]
  #v(17.2pt)

  // -- certified / approved paragraphs --
  #block(width: 100%)[
    #set par(justify: true, leading: 1.5em, spacing: 18.32pt)
    #set text(size: 12pt, hyphenate: false)
    Certified that the project work entitled #text(weight: "bold")[“#text(fill: rgb("#ff0000"))[#project_title]”] is carried out by #text(fill: purple, weight: "bold")[#author_list] bearing USNs #text(fill: purple, weight: "bold")[#usns.at(0)], #text(fill: purple, weight: "bold")[#usn_rest and #usn_last] respectively bonafide students of #text(fill: rgb("#c00000"), weight: "bold")[#college_name, #college_place] in partial fulfilment for the award of #text(fill: purple, weight: "bold")[#degree] in #text(fill: rgb("#002060"), weight: "bold")[#department] of the #text(fill: rgb("#c00000"), weight: "bold")[#university_name], #university_city during the year #academic_year. It is certified that all corrections/suggestions indicated during Internal Assessment have been incorporated in the report deposited in the departmental library.

    The project report has been approved as it satisfies the academic requirements in respect of Project work prescribed for the said Degree.
  ]

  // -- signatures --
  #v(50.4pt)
  #align(center)[
    #table(
      columns: (1fr, 1fr, 1.1fr),
      align: center,
      stroke: none,
      inset: 8pt,
      column-gutter: 20pt,
      [#line(length: 140pt, stroke: 0.6pt)], [#line(length: 140pt, stroke: 0.6pt)], [#line(length: 140pt, stroke: 0.6pt)],
      [#text(size: 12pt, weight: "bold")[Signature of the Guide]], [#text(size: 12pt, weight: "bold")[Signature of the HOD]], [#text(size: 12pt, weight: "bold")[Signature of the Principal]],
      [#text(size: 12pt, weight: "bold")[#guide]], [#text(size: 12pt, weight: "bold")[#hod]], [#text(size: 12pt, weight: "bold")[#principal]],
    )
  ]

  // -- external viva --
  #v(45pt)
  #h(180pt)#underline(text(size: 12pt, weight: "bold")[EXTERNAL VIVA])
  #v(4.9pt)
  #grid(columns: (360pt, 1fr), column-gutter: 0pt,
    align(left)[#text(size: 12pt)[Name of the Examiners]],
    align(left)[#text(size: 12pt)[Signature with date]],
  )
  #v(23.1pt)
  #grid(columns: (360pt, 1fr), column-gutter: 0pt,
    align(left)[#text(size: 12pt)[#("1" + "\u{2026}" * 11 + "." * 14)]],
    align(left)[#text(size: 12pt)[#("." * 36)]],
  )
  #v(21.6pt)
  #grid(columns: (360pt, 1fr), column-gutter: 0pt,
    align(left)[#text(size: 12pt)[#("2" + "\u{2026}" * 11 + "." * 14)]],
    align(left)[#text(size: 12pt)[#("." * 36)]],
  )
]
