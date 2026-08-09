#import "../lib.typ": *
#import "../meta.typ": *

// --- Declaration page (replicates DADS_final_edit.pdf page 3) ---
#[
  #set page(paper: "a4", margin: 0pt)
  #set text(font: "Times New Roman")
  #let _pt(v) = if type(v) == length { v } else { v * 1pt }

  // centered line placed with its glyph bbox top at y=dy
  #let tline(dy, size, weight: "regular", fill: black, edge: 0pt, body) = {
    let off = 0.77 * size - 0.1pt - edge
    place(dx: 0pt, dy: _pt(dy) + off, box(width: 100%, align(center)[
      #set text(size: size, weight: weight, fill: fill, top-edge: edge, bottom-edge: edge)
      #body
    ]))
  }

  // left-aligned line placed with its glyph bbox top at y=dy
  #let tleft(dx, dy, size, weight: "regular", fill: black, body) = {
    let off = 0.77 * size - 0.1pt
    place(dx: _pt(dx), dy: _pt(dy) + off, box[
      #set text(size: size, weight: weight, fill: fill, top-edge: 0pt, bottom-edge: 0pt)
      #body
    ])
  }

  #tline(76.2, 18pt, weight: "bold")[DECLARATION]

  // body text
  #place(dx: 50.8pt, dy: 126.11pt, box(width: 492.4pt)[
    #set text(size: 13pt, top-edge: 13.9pt, bottom-edge: 13.9pt)
    #set par(justify: true)

    We, #text(fill: rgb("#6c2c9f"))[*#author_decl_list*] students of #degree_short #semester#super[th] Semester in #department, #text(fill: rgb("#00aceb"))[*#college_name*], #college_place, hereby declare that the project work entitled #text(fill: rgb("#ff0000"))[*“#project_title”*] has been carried out by us at #college_short, #college_place, under the guidance of #text(fill: rgb("#c0504d"))[*#guide*] #guide_designation, Department of #department, #college_name, #college_place, and submitted in partial fulfillment of the requirements for the award of degree in #text(fill: rgb("#006cc0"))[*#degree in #department*] by #text(fill: rgb("#c00000"))[*#university_name*], #university_city during the academic year #academic_year_full.
  ])

  // declaration table (real Typst table, 0.75pt grid)
  #place(dx: 70.9pt, dy: 413.7pt, table(
    columns: (177pt, 109.4pt, 168.9pt),
    rows: (26.3pt, 33.6pt, 33.6pt, 33.6pt, 33.6pt),
    stroke: 0.75pt + black,
    inset: (x: 8pt, y: 3.4pt),
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
      [
        #set text(12pt)
        #align(left)[#a.name]
      ],
      [
        #set text(12pt)
        #align(left)[#a.usn]
      ],
      [],
    )).flatten(),
  ))

  // date and place
  #tleft(72.1, 635.2, 12pt)[Date:]
  #tleft(72.1, 654.2, 12pt)[Place: #college_place]
]
