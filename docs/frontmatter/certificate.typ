#import "../lib.typ": *
#import "../meta.typ": *

// --- Certificate page (replicates DADS_final_edit.pdf page 2) ---
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

  #tline(48.6, 14pt, weight: "bold", fill: rgb("#c00000"))[#college_upper]
  #tline(66.4, 10pt)[#college_unit]
  #tline(79.8, 9pt, fill: rgb("#0000ff"))[#affiliation]
  #tline(91.5, 12pt)[#college_address]
  #tline(105.2, 11pt, weight: "bold", fill: rgb("#006cc0"))[#dept_block1]
  #tline(118.0, 11pt, weight: "bold", fill: rgb("#006cc0"))[#dept_block2]
  #place(dx: 264.0pt, dy: 140.7pt, image("../assets/vcet_logo.png", width: 76.2pt, height: 56.1pt))
  #tline(205.8, 18pt, weight: "bold")[CERTIFICATE]

  // body text
  #place(dx: 54pt, dy: 236.9pt, box(width: 495pt)[
    #set text(size: 13pt, top-edge: 13.9pt, bottom-edge: 13.9pt)
    #set par(justify: true, spacing: 9pt)

    Certified that the project work entitled #text(fill: rgb("#ff0000"))[*“#project_title”*] is carried out by #text(fill: rgb("#6c2c9f"))[*#author_list*] bearing USNs #text(fill: rgb("#6c2c9f"))[*#usns.at(0)*], #text(fill: rgb("#6c2c9f"))[*#usn_rest and #usn_last*] respectively bonafide students of #text(fill: rgb("#c00000"))[*#college_name, #college_place*] in partial fulfilment for the award of #text(fill: rgb("#6c2c9f"))[*#degree*] in #text(fill: rgb("#6c2c9f"))[*#department*] of the #text(fill: rgb("#c00000"))[*#university_name, #university_city*] during the year #academic_year.

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
  #sigcol(59.5pt, 132.5pt, "________________________", [Signature of Guide], [#guide])
  #sigcol(236.1pt, 132pt, "________________________", [Signature of Project Coordinator], [#guide])
  #sigcol(410.0pt, 137.5pt, "_________________________", [Signature of HOD], [#hod])
]
