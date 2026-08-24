#import "../lib.typ": *
#import "../meta.typ": *

// --- Declaration ---
#[
  #set page(paper: "a4", margin: (top: 59pt, bottom: 54pt, left: 90pt, right: 72pt))
  #set text(font: "Times New Roman")
  #set par(leading: 0pt, spacing: 0pt)

  #align(center, text(size: 18pt, weight: "bold")[DECLARATION])
  #v(38.0pt)

  // -- declaration body --
  #block(width: 100%)[
    #set par(justify: true, leading: 16.42pt, spacing: 16.42pt)
    #set text(size: 12pt, hyphenate: false)
    #text(weight: "bold")[We], #text(fill: rgb("#7030a0"), weight: "bold")[#author_decl_list] students of #degree_short #semester#super[th] Semester in #department, #text(fill: rgb("#00b0f0"), weight: "bold")[#college_name], #college_place, hereby declare that the project work entitled #text(weight: "bold")[“#text(fill: rgb("#ff0000"))[#project_title]”] has been carried out by us at #college_short, #college_place, under the guidance of #text(fill: rgb("#c0504d"), weight: "bold")[#guide,] #guide_designation, Department of #department, #college_name, #college_place, and submitted in partial fulfilment of the requirements for the award of degree in #text(fill: rgb("#0070c0"), weight: "bold")[#degree in #department] by #text(fill: rgb("#c00000"), weight: "bold")[#university_name], #university_city during the academic year #academic_year_full.
  ]

  // -- signature table --
  #v(59.8pt)
  #move(dx: 16.6pt)[
    #table(
      columns: (162pt, 94.5pt, 154.1pt),
      rows: (25.9pt, 41pt, 41pt, 41pt, 41pt),
      stroke: 0.6pt,
      inset: (y: 4pt),
      align: center + horizon,
      text(size: 12pt)[Name of the students],
      text(size: 12pt)[USN],
      text(size: 12pt)[Signature with date],
      table.cell()[], table.cell()[], table.cell()[],
      table.cell()[], table.cell()[], table.cell()[],
      table.cell()[], table.cell()[], table.cell()[],
      table.cell()[], table.cell()[], table.cell()[],
    )
  ]

  // -- date / place --
  #v(45.0pt)
  #set text(size: 12pt)
  Date:
  #v(16.2pt)
  Place: #college_place
]
