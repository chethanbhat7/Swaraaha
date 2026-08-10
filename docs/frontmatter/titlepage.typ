#import "../lib.typ": *
#import "../meta.typ": *

// --- Title page ---
// All spacing is relative: every element is its own block and the gaps between
// them are taken from DADS_final_edit.pdf. If the project title wraps to more
// lines, the rest of the page simply flows down and nothing overlaps.
#let title_guide = "Prof. AJAY SHASTRY C.G."

// One row of the submitted-by table: a spacer column pushes the name and USN
// columns to their DADS x positions.
#let _name_row(name, usn) = grid(
  columns: (63.1pt, 287.4pt, 65.28pt),
  column-gutter: 0pt,
  [],
  text(weight: "bold", fill: rgb("#006cc0"))[#upper(name)],
  align(right, text(weight: "bold", fill: rgb("#006cc0"))[#usn]),
)

#[
  #set page(paper: "a4", margin: (top: 0pt, bottom: 0pt, left: 58.9pt, right: 41.3pt))
  #set text(font: "Times New Roman", size: 12pt)
  #set par(spacing: 5.92pt, leading: 5.92pt)

  // -- university header --
  #v(53.23pt)
  #align(center, text(weight: "bold", fill: rgb("#c00000"))[#university_title])
  #align(center, text(weight: "bold")[#university_address])

  // -- vtu logo --
  #v(1.48pt)
  #align(center, image("../assets/vtu_logo.png", width: 104.30pt, height: 108.55pt, fit: "stretch"))
  #v(-4.95pt)

  // -- report type / On / project title --
  #v(0.76pt)
  #align(center, text(weight: "bold")[#report_type])
  #v(4.05pt)
  #align(center, text(weight: "bold")[On])
  #v(3.60pt)
  #align(center, text(weight: "bold", fill: rgb("#ff0000"))[“#project_title”])
  #v(9.95pt)
  #align(center, text(weight: "bold")[Submitted in partial fulfilment of the requirements for the award of])
  #v(4.70pt)
  #align(center, text(weight: "bold")[#degree_upper])
  #v(4.90pt)
  #align(center, text(weight: "bold", fill: rgb("#6c2c9f"))[#department_upper])

  // -- submitted by + name table --
  #v(14.50pt)
  #align(center, text(weight: "bold")[Submitted By])
  #v(3.05pt)
  #grid(columns: (63.1pt, 287.4pt, 65.28pt), column-gutter: 0pt,
    [],
    [#h(27.85pt)#text(weight: "bold", fill: rgb("#001f5f"))[Name]],
    [#h(15.5pt)#text(weight: "bold", fill: rgb("#001f5f"))[USN]],
  )
  #v(2.90pt)
  #_name_row(authors.at(0).name, authors.at(0).usn)
  #v(2.90pt)
  #_name_row(authors.at(1).name, authors.at(1).usn)
  #v(3.70pt)
  #_name_row(authors.at(2).name, authors.at(2).usn)
  #v(3.05pt)
  #_name_row(authors.at(3).name, authors.at(3).usn)

  // -- guidance --
  #v(13.30pt)
  #align(center, text(weight: "bold")[Under the Guidance of])
  #v(-0.20pt)
  #align(center, text(weight: "bold", fill: rgb("#c00000"))[#title_guide])
  #v(-0.05pt)
  #align(center, text(weight: "bold")[Assistant Professor])

  // -- vcet logo --
  #v(9.62pt)
  #align(center, image("../assets/vcet_logo.png", width: 114.20pt, height: 83.90pt))
  #v(7.73pt)
  #h(29.25pt)#box(width: 437.1pt, height: 2.3pt, fill: black)
  #v(10.67pt)
  #align(center, text(weight: "bold", fill: rgb("#006cc0"))[#dept_block1])
  #v(1.80pt)
  #align(center, text(weight: "bold", fill: rgb("#006cc0"))[#dept_block2])
  #v(2.10pt)
  #align(center, text(weight: "bold", fill: rgb("#c00000"))[#college_block1])
  #v(1.95pt)
  #align(center, text(weight: "bold", fill: rgb("#c00000"))[#college_block2])
  #v(0.41pt)
  #align(center)[#college_unit]
  #v(12.86pt)
  #align(center, text(fill: rgb("#0000ff"))[#affiliation_line1])
  #v(-0.44pt)
  #align(center, text(fill: rgb("#0000ff"))[#affiliation_line2])
  #v(0.81pt)
  #align(center)[#college_address]
  #v(2.10pt)
  #align(center, text(weight: "bold", fill: rgb("#ff0000"))[#submission_month])
]
