#import "lib.typ": *

#set text(size: 12pt)
#set par(justify: true, leading: 18pt, spacing: 18pt)
#set par(justify: false)

#pagebreak()

// Heading for the TOC; rendered invisibly so the visible title below is centered like other chapter heads
#[
  #set text(size: 0pt)
  #v(-0.8em)
  #heading(level: 1, numbering: none)[PERSONAL PROFILE]
]
#align(center)[#text(size: 18pt, weight: "bold")[PERSONAL PROFILE]]

#v(2em)

#let profile_photo(path) = box(
  image(path, width: 100pt, height: 125pt, fit: "cover"),
  stroke: 1pt + black,
  radius: 2pt,
)

#let student_details(name, usn, email, phone) = [
  #text(weight: "bold")[#name]
  #v(0.2em)
  USN: #usn
  #v(0.2em)
  Email ID: #email
  #v(0.2em)
  Phone: #phone
]

#table(
  columns: (110pt, 1fr),
  stroke: 0.5pt + black,
  inset: 10pt,
  align: left,
  align(center)[#profile_photo("/assets/Ajay Shastry C G.jpg")],
  [
    #text(weight: "bold")[Prof. Ajay Shastry C G]
    #v(0.4em)
    He has a teaching experience of 8 years. His areas of interest include
    Cybersecurity, AI/ML, and Internet of Things.
    #v(0.5em)
    Email ID: #text("ajayshastrycg.ai@vcetputtur.ac.in")
    #v(0.2em)
    Phone: #text[+91 99021 94905]
  ],
  align(center)[#profile_photo("/assets/K Shreekrishna Upadhyaya.jpeg")],
  student_details(
    "K Shreekrishna Upadhyaya",
    "4VP23AI020",
    "upadhyayashreekrishna@gmail.com",
    "+91 96861 08613",
  ),
  align(center)[#profile_photo("/assets/M Chethan Keshav Bhat.jpeg")],
  student_details(
    "M Chethan Keshav Bhat",
    "4VP23AI023",
    "chethanbhat777@gmail.com",
    "+91 80734 68867",
  ),
  align(center)[#profile_photo("/assets/Skanda Prasad K.jpeg")],
  student_details(
    "Skanda Prasad K",
    "4VP23AI051",
    "skandaprasad02@gmail.com",
    "+91 87628 89622",
  ),
  align(center)[#profile_photo("/assets/Srinivas Hegde M.jpeg")],
  student_details(
    "Srinivas Hegde M",
    "4VP23AI054",
    "hegdesrinivasm@gmail.com",
    "+91 90601 59605",
  ),
)