#import "../lib.typ": *

// --- Certificate page (replicates DADS_final_edit.pdf page 2) ---
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

  #tline(48.6, 14pt, weight: "bold", fill: rgb("#c00000"))[VIVEKANANDA COLLEGE OF ENGINEERING & TECHNOLOGY]
  #tline(66.4, 10pt)[[A Unit of Vivekananda Vidyavardhaka Sangha Puttur (R)]]
  #tline(79.8, 9pt, fill: rgb("#0000ff"))[Affiliated to Visvesvaraya Technological University and Approved by AICTE New Delhi & Govt. of Karnataka]
  #tline(91.5, 12pt)[Nehru Nagar, Puttur - 574203, DK, Karnataka, India]
  #tline(105.2, 11pt, weight: "bold", fill: rgb("#006cc0"))[DEPARTMENT OF ARTIFICIAL INTELLIGENCE & MACHINE]
  #tline(118.0, 11pt, weight: "bold", fill: rgb("#006cc0"))[LEARNING]
  #place(dx: 264.0pt, dy: 140.7pt, image("../assets/vcet_logo.png", width: 76.2pt, height: 56.1pt))
  #tline(205.8, 18pt, weight: "bold")[CERTIFICATE]

  // body text
  #place(dx: 54pt, dy: 236.9pt, box(width: 495pt)[
    #set text(size: 13pt, top-edge: 13.9pt, bottom-edge: 13.9pt)
    #set par(justify: true, spacing: 9pt)

    Certified that the mini project work entitled #text(fill: rgb("#ff0000"))[*“Stutter Detection and Classification System for Speech Pathology Assistance”*] is carried out by #text(fill: rgb("#6c2c9f"))[*Mr. K Shreekrishna Upadhyaya, Mr. M Chethan Keshav Bhat, Mr. Skanda Prasad K, Mr. Srinivas Hegde M*] bearing USNs #text(fill: rgb("#6c2c9f"))[*4VP23AI020*], #text(fill: rgb("#6c2c9f"))[*4VP23AI023, 4VP23AI051 and 4VP23AI054*] respectively bonafide students of #text(fill: rgb("#c00000"))[*Vivekananda College of Engineering & Technology, Puttur*] in partial fulfilment for the award of #text(fill: rgb("#6c2c9f"))[*Bachelor of Engineering*] in #text(fill: rgb("#6c2c9f"))[*Artificial Intelligence & Machine Learning*] of the #text(fill: rgb("#c00000"))[*Visvesvaraya Technological University, Belagavi*] during the year 2025-26.

    It is certified that all corrections/suggestions indicated during Internal Assessment have been incorporated in the report deposited in the departmental library.

    The project report has been approved as it satisfies the academic requirements in respect of Project work prescribed for the said Degree.
  ])

  // signature block
  #let sigcol(dx, width, uscore, label, name) = {
    place(dx: dx, dy: 657.5pt + 8.37pt, box(width: width, align(center)[
      #set text(size: 11pt, top-edge: 0pt, bottom-edge: 0pt)
      #uscore
    ]))
    place(dx: dx, dy: 681.2pt + 4.46pt, box(width: width, align(center)[
      #set text(size: 13pt, top-edge: 5.45pt, bottom-edge: 5.45pt)
      #label
    ]))
    place(dx: dx, dy: 709.1pt + 4.46pt, box(width: width, align(center)[
      #set text(size: 13pt, weight: "bold", top-edge: 5.45pt, bottom-edge: 5.45pt)
      #name
    ]))
  }
  #sigcol(59.5pt, 132.5pt, "________________________", [Signature of Guide], [Prof. Ajay Shastry C G])
  #sigcol(236.1pt, 132pt, "________________________", [Signature of Mini project Coordinator], [Prof. Ajay Shastry C G])
  #sigcol(410.0pt, 137.5pt, "_________________________", [Signature of HOD], [Prof. Abhishek Kumar K])
]
