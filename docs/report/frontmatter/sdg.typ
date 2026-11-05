// --- Sustainable Development Goals (SDGs) ---
#import "../meta.typ": project_title

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.3cm),
)

#set text(font: "Libertinus Serif", size: 12pt, hyphenate: false)
#set par(justify: true, leading: 1em, spacing: 0pt)



#align(center)[
  #text(size: 16pt, weight: "bold")[
    #underline[SUSTAINABLE DEVELOPMENT GOALS (SDGs)]
  ]
]

// Block gaps are set explicitly so the page lines up with the reference layout.
#v(22.8pt)

#text(weight: "bold")[Project Title: #project_title]

#v(19.6pt)

Identify the relevant Sustainable Development Goal(s) (SDGs) associated with
the project and justify how each selected goal aligns with the project's
objectives, implementation, and outcomes.

#v(19.6pt)

// --- Table ---
#set par(justify: true, leading: 0.8em, spacing: 0pt)

#let sdg_logo(file) = align(center + top)[
  #image("/assets/SDG LOGOS/" + file, width: 2.25cm)
]

#let sdg_head(body) = align(center)[
  #set par(justify: false, leading: 0pt)
  #text(size: 11pt, weight: "bold")[#body]
]

#table(
  columns: (2.7cm, 4.35cm, 1fr),
  stroke: 0.5pt + black,
  inset: (x: 5pt, y: 4pt),
  align: (center + top, left + top, left + top),
  table.header(
    sdg_head[Goal No],
    sdg_head[SDG -- Description],
    sdg_head[Justification],
  ),
  sdg_logo("SDG3.png"),
  [
    #text(size: 10.5pt)[*Good Health and Well-being.* Ensure healthy lives and
      promote well-being for all at all ages.]
  ],
  [
    #text(size: 10.5pt)[Stuttering affects communication, mental health, and
      social participation. Automated speech assessment supports early
      detection, objective monitoring, and timely intervention, making
      consistent care available beyond specialist centres. This advances
      Target 3.4 (mental health and well-being) and Target 3.8 (universal
      health coverage).]
  ],

  sdg_logo("SDG4.png"),
  [
    #text(size: 10.5pt)[*Quality Education.* \
      Ensure inclusive and equitable quality education and promote lifelong
      learning opportunities for all.]
  ],
  [
    #text(size: 10.5pt)[Children and students who stutter often hesitate to
      speak in class, limiting participation and learning. An accessible
      screening tool helps schools and therapists identify affected learners
      early and provide suitable therapy and classroom support. This aligns
      with Target 4.5 (equal access for vulnerable groups) and Target 4.a
      (inclusive learning environments).]
  ],

  sdg_logo("SDG9.png"),
  [
    #text(size: 10.5pt)[*Industry, Innovation and Infrastructure.* \
      Build resilient infrastructure, promote inclusive and sustainable
      industrialization and foster innovation.]
  ],
  [
    #text(size: 10.5pt)[Applying deep learning to a low-resource clinical
      speech problem demonstrates affordable innovation in digital health.
      The work creates a reusable speech-analytics pipeline and supports
      Target 9.5 (enhance research and technological capabilities).]
  ],

  sdg_logo("SDG10.png"),
  [
    #text(size: 10.5pt)[*Reduced Inequalities.* \
      Reduce inequality within and among countries.]
  ],
  [
    #text(size: 10.5pt)[Certificate images]
  ],
  colbreak(),
  align(center, image("/assets/JS-Shreekrishna.png", width: 3cm)),
  colbreak(),
  align(center, image("/assets/JS-chethan.jpeg", width: 3cm)),
  colbreak(),
  align(center, image("/assets/JS-skanda.jpeg", width: 3cm)),
  colbreak(),
  align(center, image("/assets/JS-Srinivas.jpeg", width: 3cm))
)
