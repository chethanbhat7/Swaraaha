#import "../lib.typ": *
#import "../meta.typ": *

// --- Title page (replicates DADS_final_edit.pdf page 1) ---
#[
  #set page(paper: "a4", margin: 0pt)
  #set text(font: "Times New Roman")
  #let _pt(v) = if type(v) == length { v } else { v * 1pt }

  // room left for the project title (wraps to two lines); elements below shift by S
  #let S = 28

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

  #let row_ys = (352.2, 369.1, 386.9, 404.0)

  #tline(52.3, 12pt, weight: "bold", fill: rgb("#c00000"))[#university_title]
  #tline(66.3, 12pt, weight: "bold")[#university_address]
  #place(dx: 249.9pt, dy: 82.8pt, image("../assets/vtu_logo.png", width: 104.35pt, height: 108.6pt, fit: "stretch"))
  #tline(192.1, 12pt, weight: "bold")[#report_type]
  #tline(210.2, 12pt, weight: "bold")[On]
  #tline(227.9, 12pt, weight: "bold", fill: rgb("#ff0000"), edge: 5pt)[“#project_title_break”]
  #tline(251.9 + S, 12pt, weight: "bold")[Submitted in partial fulfilment of the requirements for the award of]
  #tline(270.6 + S, 12pt, weight: "bold")[#degree_upper]
  #tline(289.6 + S, 12pt, weight: "bold", fill: rgb("#6c2c9f"))[#department_upper]
  #tline(318.1 + S, 12pt, weight: "bold")[Submitted By]
  #tleft(149.9, 335.2 + S, 12pt, weight: "bold", fill: rgb("#001f5f"))[Name]
  #tleft(424.9, 335.2 + S, 12pt, weight: "bold", fill: rgb("#001f5f"))[USN]
  #for (i, a) in authors.enumerate() {
    tleft(122.0, row_ys.at(i) + S, 12pt, weight: "bold", fill: rgb("#006cc0"))[#upper(a.name)]
    tleft(409.0, row_ys.at(i) + S, 12pt, weight: "bold", fill: rgb("#006cc0"))[#a.usn]
  }
  #tline(431.3 + S, 12pt, weight: "bold")[Under the Guidance of]
  #tline(445.2 + S, 12pt, weight: "bold", fill: rgb("#c00000"))[#upper(guide)]
  #tline(459.2 + S, 12pt, weight: "bold")[Assistant Professor]
  #place(dx: 244.9pt, dy: _pt(483.7 + S), image("../assets/vcet_logo.png", width: 114.3pt, height: 84.0pt))
  #place(dx: 88.15pt, dy: _pt(581.35 + S), box(width: 437.1pt, height: 2.3pt, fill: black))
  #tline(599.3 + S, 12pt, weight: "bold", fill: rgb("#006cc0"))[#dept_block1]
  #tline(615.1 + S, 12pt, weight: "bold", fill: rgb("#006cc0"))[#dept_block2]
  #tline(631.3 + S, 12pt, weight: "bold", fill: rgb("#c00000"))[#college_block1]
  #tline(647.3 + S, 12pt, weight: "bold", fill: rgb("#c00000"))[#college_block2]
  #tline(661.9 + S, 12pt)[#college_unit]
  #tline(689.0 + S, 12pt, fill: rgb("#0000ff"))[#affiliation_line1]
  #tline(702.76 + S, 12pt, fill: rgb("#0000ff"))[#affiliation_line2]
  #tline(717.8 + S, 12pt)[#college_address]
  #tline(734.0 + S, 12pt, weight: "bold", fill: rgb("#ff0000"))[#submission_month]
]
