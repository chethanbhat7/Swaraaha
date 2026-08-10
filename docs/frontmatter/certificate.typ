#import "../lib.typ": *
#import "../meta.typ": *

#[
  #set page(paper: "a4", margin: (left: 54pt, right: 46.3pt, top: 49.74pt, bottom: 60pt))
  #set text(font: "Times New Roman")

  #front_center_line(14pt, weight: "bold", fill: rgb("#c00000"))[#college_upper]
  #v(-8.92pt)
  #front_center_line(10pt)[#college_unit]
  #v(-5.61pt)
  #front_center_line(9pt, fill: rgb("#0000ff"))[#affiliation]
  #v(-8.73pt)
  #front_center_line(12pt)[#college_address]
  #v(-8.96pt)
  #front_center_line(11pt, weight: "bold", fill: rgb("#006cc0"))[#dept_block1]
  #v(-7.85pt)
  #front_center_line(11pt, weight: "bold", fill: rgb("#006cc0"))[#dept_block2]
  #v(0pt)
  #align(center, image("../assets/vcet_logo.png", height: 56.1pt))
  #v(-11.19pt)
  #front_center_line(18pt, weight: "bold")[CERTIFICATE]
  #v(0.77pt)

  #set par(justify: true, spacing: 15.39pt, leading: 13.39pt)
  #set text(size: 13pt)

  Certified that the project work entitled #text(fill: rgb("#ff0000"))[*“#project_title”*] is carried out by #text(fill: rgb("#6c2c9f"))[*#author_list*] bearing USNs #text(fill: rgb("#6c2c9f"))[*#usns.at(0)*], #text(fill: rgb("#6c2c9f"))[*#usn_rest and #usn_last*] respectively bonafide students of #text(fill: rgb("#c00000"))[*#college_name, #college_place*] in partial fulfilment for the award of #text(fill: rgb("#6c2c9f"))[*#degree*] in #text(fill: rgb("#6c2c9f"))[*#department*] of the #text(fill: rgb("#c00000"))[*#university_name, #university_city*] during the year #academic_year.

  It is certified that all corrections/suggestions indicated during Internal Assessment have been incorporated in the report deposited in the departmental library.

  The project report has been approved as it satisfies the academic requirements in respect of project work prescribed for the said Degree.

  #set par(leading: 0.65em, spacing: 2pt)

  #v(153.25pt)

  #signature_block((
    ("Signature of Guide", guide, 17.0pt, 0.65em, 132pt),
    ([Signature of \ Project Coordinator], project_coordinator, 3.04pt, 4.94pt, 132pt),
    ("Signature of HOD", hod, 17.0pt, 0.65em, 137.5pt),
  ), columns: (1fr, 1.4fr, 1fr), gutter: 1.5em, line_gap: 10.42pt)
]
