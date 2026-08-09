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

  // centered within a fixed-width box placed with glyph bbox top at y=dy
  #let tcenter(dx, width, dy, size, body) = {
    let off = 0.77 * size - 0.1pt
    place(dx: _pt(dx), dy: _pt(dy) + off, box(width: width, align(center)[
      #set text(size: size, top-edge: 0pt, bottom-edge: 0pt)
      #body
    ]))
  }

  #tline(76.2, 18pt, weight: "bold")[DECLARATION]

  // body text
  #place(dx: 50.8pt, dy: 126.11pt, box(width: 492.4pt)[
    #set text(size: 13pt, top-edge: 13.9pt, bottom-edge: 13.9pt)
    #set par(justify: true)

    We, #text(fill: rgb("#6c2c9f"))[*#author_decl_list*] students of #degree_short #semester#super[th] Semester in #department, #text(fill: rgb("#00aceb"))[*#college_name*], #college_place, hereby declare that the project work entitled #text(fill: rgb("#ff0000"))[*“#project_title”*] has been carried out by us at #college_short, #college_place, under the guidance of #text(fill: rgb("#c0504d"))[*#guide*] #guide_designation, Department of #department, #college_name, #college_place, and submitted in partial fulfillment of the requirements for the award of degree in #text(fill: rgb("#006cc0"))[*#degree in #department*] by #text(fill: rgb("#c00000"))[*#university_name*], #university_city during the academic year #academic_year_full.
  ])

  // table grid (0.75pt)
  #place(dx: 70.9pt, dy: 413.7pt, box(width: 455.3pt, height: 0.75pt, fill: black))
  #place(dx: 70.9pt, dy: 440.0pt, box(width: 455.3pt, height: 0.75pt, fill: black))
  #place(dx: 70.9pt, dy: 473.5pt, box(width: 455.3pt, height: 0.75pt, fill: black))
  #place(dx: 70.9pt, dy: 507.3pt, box(width: 455.3pt, height: 0.75pt, fill: black))
  #place(dx: 70.9pt, dy: 541.3pt, box(width: 455.3pt, height: 0.75pt, fill: black))
  #place(dx: 70.9pt, dy: 574.9pt, box(width: 455.3pt, height: 0.75pt, fill: black))
  #place(dx: 70.9pt, dy: 413.7pt, box(width: 0.75pt, height: 161.2pt, fill: black))
  #place(dx: 247.9pt, dy: 413.7pt, box(width: 0.75pt, height: 161.2pt, fill: black))
  #place(dx: 357.3pt, dy: 413.7pt, box(width: 0.75pt, height: 161.2pt, fill: black))
  #place(dx: 526.2pt, dy: 413.7pt, box(width: 0.75pt, height: 161.2pt, fill: black))

  // table header
  #tcenter(70.9pt, 177pt, 416.3, 12pt)[Name of the students]
  #tcenter(247.9pt, 109.4pt, 416.3, 12pt)[USN]
  #tcenter(357.3pt, 168.9pt, 416.3, 12pt)[Signature with date]

  // table data
  #let row_ys = (449.8, 483.6, 517.6, 551.2)
  #for (i, a) in authors.enumerate() {
    tleft(95.4, row_ys.at(i), 12pt)[#a.name]
    tleft(271.8, row_ys.at(i), 12pt)[#a.usn]
  }

  // date and place
  #tleft(72.1, 635.2, 12pt)[Date:]
  #tleft(72.1, 654.2, 12pt)[Place: #college_place]
]
