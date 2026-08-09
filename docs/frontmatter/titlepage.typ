#import "../lib.typ": *

// --- Title page (replicates DADS_final_edit.pdf page 1) ---
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

  #tline(52.3, 12pt, weight: "bold", fill: rgb("#c00000"))[VISVESVARAYA TECHNOLOGICAL UNIVERSITY]
  #tline(66.3, 12pt, weight: "bold")[JNANA SANGAMA, BELAGAVI– 590018, KARNATAKA, INDIA]
  #place(dx: 249.9pt, dy: 82.8pt, image("../assets/vtu_logo.png", width: 104.35pt, height: 108.6pt, fit: "stretch"))
  #tline(192.1, 12pt, weight: "bold")[A MINI PROJECT REPORT]
  #tline(210.2, 12pt, weight: "bold")[On]
  #tline(227.9, 12pt, weight: "bold", fill: rgb("#ff0000"))[“Stutter Detection and Classification System for Speech Pathology Assistance”]
  #tline(251.9, 12pt, weight: "bold")[Submitted in partial fulfilment of the requirements for the award of]
  #tline(270.6, 12pt, weight: "bold")[BACHELOR OF ENGINEERING]
  #tline(289.6, 12pt, weight: "bold", fill: rgb("#6c2c9f"))[ARTIFICIAL INTELLIGENCE & MACHINE LEARNING]
  #tline(318.1, 12pt, weight: "bold")[Submitted By]
  #tleft(149.9, 335.2, 12pt, weight: "bold", fill: rgb("#001f5f"))[Name]
  #tleft(424.9, 335.2, 12pt, weight: "bold", fill: rgb("#001f5f"))[USN]
  #tleft(122.0, 352.2, 12pt, weight: "bold", fill: rgb("#006cc0"))[K SHREEKRISHNA UPADHYAYA]
  #tleft(409.0, 352.2, 12pt, weight: "bold", fill: rgb("#006cc0"))[4VP23AI020]
  #tleft(122.0, 369.1, 12pt, weight: "bold", fill: rgb("#006cc0"))[M CHETHAN KESHAV BHAT]
  #tleft(409.0, 369.1, 12pt, weight: "bold", fill: rgb("#006cc0"))[4VP23AI023]
  #tleft(122.0, 386.9, 12pt, weight: "bold", fill: rgb("#006cc0"))[SKANDA PRASAD K]
  #tleft(409.0, 386.9, 12pt, weight: "bold", fill: rgb("#006cc0"))[4VP23AI051]
  #tleft(122.0, 404.0, 12pt, weight: "bold", fill: rgb("#006cc0"))[SRINIVAS HEGDE M]
  #tleft(409.0, 404.0, 12pt, weight: "bold", fill: rgb("#006cc0"))[4VP23AI054]
  #tline(431.3, 12pt, weight: "bold")[Under the Guidance of]
  #tline(445.2, 12pt, weight: "bold", fill: rgb("#c00000"))[Prof. AJAY SHASTRY C.G.]
  #tline(459.2, 12pt, weight: "bold")[Assistant Professor]
  #place(dx: 244.9pt, dy: 483.7pt, image("../assets/vcet_logo.png", width: 114.3pt, height: 84.0pt))
  #place(dx: 88.15pt, dy: 581.35pt, box(width: 437.1pt, height: 2.3pt, fill: black))
  #tline(599.3, 12pt, weight: "bold", fill: rgb("#006cc0"))[DEPARTMENT OF ARTIFICIAL INTELLIGENCE & MACHINE]
  #tline(615.1, 12pt, weight: "bold", fill: rgb("#006cc0"))[LEARNING]
  #tline(631.3, 12pt, weight: "bold", fill: rgb("#c00000"))[VIVEKANANDA COLLEGE OF ENGINEERING &]
  #tline(647.3, 12pt, weight: "bold", fill: rgb("#c00000"))[TECHNOLOGY]
  #tline(661.9, 12pt)[[A Unit of Vivekananda Vidyavardhaka Sangha Puttur (R)]]
  #tline(689.0, 12pt, fill: rgb("#0000ff"))[Affiliated to Visvesvaraya Technological University and Approved by AICTE New Delhi & Govt., of]
  #tline(702.76, 12pt, fill: rgb("#0000ff"))[Karnataka]
  #tline(717.8, 12pt)[Nehru Nagar, Puttur - 574 203, DK, Karnataka, India.]
  #tline(734.0, 12pt, weight: "bold", fill: rgb("#ff0000"))[NOVEMBER 2025]
]
