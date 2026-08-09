#import "../lib.typ": *

// --- Declaration page (replicates DADS_final_edit.pdf page 3) ---
#[
  #set page(paper: "a4", margin: 0pt)
  #set text(font: "Times New Roman")
  #let _pt(v) = if type(v) == length { v } else { v * 1pt }

  // centered line placed with its glyph bbox top at y=dy
  #let tline(dy, size, weight: "regular", fill: black, edge: 0pt, body) = {
    let off = if edge > 0pt { -3.99pt } else { 0.77 * size - 0.1pt }
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

    We, #text(fill: rgb("#6c2c9f"))[*Mr. K Shreekrishna Upadhyaya (4VP23AI020), Mr. M Chethan Keshav Bhat (4VP23AI023), Mr. Skanda Prasad K (4VP23AI051), Mr. Srinivas Hegde M (4VP23AI054)*] students of B.E. 5#super[th] Semester in Artificial Intelligence & Machine Learning, #text(fill: rgb("#00aceb"))[*Vivekananda College of Engineering & Technology*], Puttur, hereby declare that the project work entitled #text(fill: rgb("#ff0000"))[*“Stutter Detection and Classification System for Speech Pathology Assistance”*] has been carried out by us at VCET, Puttur, under the guidance of #text(fill: rgb("#c0504d"))[*Prof. Ajay shastry C G*] Assistant Professor, Department of Artificial Intelligence & Machine Learning, Vivekananda College of Engineering & Technology, Puttur, and submitted in partial fulfillment of the requirements for the award of degree in #text(fill: rgb("#006cc0"))[*Bachelor of Engineering in Artificial Intelligence & Machine Learning*] by #text(fill: rgb("#c00000"))[*Visvesvaraya Technological University*], Belagavi during the academic year 2025-2026.
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
  #tleft(95.4, 449.8, 12pt)[K Shreekrishna Upadhyaya]
  #tleft(271.8, 449.8, 12pt)[4VP23AI020]
  #tleft(95.4, 483.6, 12pt)[M Chethan Keshav Bhat]
  #tleft(271.8, 483.6, 12pt)[4VP23AI023]
  #tleft(95.4, 517.6, 12pt)[Skanda Prasad K]
  #tleft(271.8, 517.6, 12pt)[4VP23AI051]
  #tleft(95.4, 551.2, 12pt)[Srinivas Hegde M]
  #tleft(271.8, 551.2, 12pt)[4VP23AI054]

  // date and place
  #tleft(72.1, 635.2, 12pt)[Date:]
  #tleft(72.1, 654.2, 12pt)[Place: Puttur]
]
