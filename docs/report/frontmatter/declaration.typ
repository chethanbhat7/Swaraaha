#import "../lib.typ": *
#import "../meta.typ": *

#[
  #set page(paper: "a4", margin: (top: 77.62pt, bottom: 2.5cm, left: 50.8pt, right: 52.08pt))
  #set text(font: "Times New Roman")

  #front_center_line(18pt, weight: "bold")[DECLARATION]
  #v(19.56pt)

  #set par(justify: true, leading: 13.39pt, spacing: 0pt)
  #set text(size: 13pt)

  We, #text(fill: rgb("#6c2c9f"))[*#author_decl_list*] students of #degree_short #semester#super[th] Semester in #department, #text(fill: rgb("#00aceb"))[*#college_name*], #college_place, hereby declare that the project work entitled #text(fill: rgb("#ff0000"))[*“#project_title”*] has been carried out by us at #college_short, #college_place, under the guidance of #text(fill: rgb("#c0504d"))[*#guide*] #guide_designation, Department of #department, #college_name, #college_place, and submitted in partial fulfillment of the requirements for the award of degree in #text(fill: rgb("#006cc0"))[*#degree in #department*] by #text(fill: rgb("#c00000"))[*#university_name*], #university_city during the academic year #academic_year_full.

  #set par(spacing: 0pt, leading: 0.65em)
  #v(50.1pt)

  #grid(columns: (20.55pt, 454.45pt), column-gutter: 0pt)[
  ][
    #table(
      columns: (176.3pt, 113.9pt, 164.25pt),
      rows: (26.3pt, 33.6pt, 33.6pt, 33.6pt, 33.6pt),
      stroke: 0.75pt + black,
      inset: (x: 24.1pt, y: 3.4pt),
      align: (center, center, center),
      [
        #set text(12pt)
        Name of the students
      ],
      [
        #set text(12pt)
        USN
      ],
      [
        #set text(12pt)
        Signature with date
      ],
      ..authors.map(a => (
        table.cell(
          inset: (left: 24.1pt, right: 19.5pt, top: 3.4pt, bottom: 3.4pt),
        )[
          #set text(12pt)
          #v(7.19pt)
          #align(left)[#a.name]
        ],
        [
          #set text(12pt)
          #v(7.19pt)
          #align(left)[#a.usn]
        ],
        [],
      )).flatten(),
    )
  ]

  #v(61.6pt)

  #set text(size: 12pt)
  #h(21.3pt)Date:
  #v(10.68pt)
  #h(21.3pt)Place: #college_place
]
