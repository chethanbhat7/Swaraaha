#let lit_survey_counter = counter("literature counter")

#let non_outlined_heading(level: 1, body) = {
  align(center)[#heading(level: level, outlined: false)[#body]]
  v(0.5em)
}

#let chapter_heading(body) = {
  counter(heading).step()
  counter(figure.where(kind: table)).update(0)

  context {
    let chap = counter(heading).get().first()
    let toc_title = [CHAPTER #chap #body]

    // tricks
    [
      #set text(size: 0pt)
      #v(-0.8em)
      #heading(numbering: none)[#toc_title]
    ]

    box(width: 100%, inset: 0pt)[
      #set text(weight: "bold", size: 16pt)
      #set align(left)
      CHAPTER #chap
      #set text(size: 18pt)
      #set align(center)
      #v(-0.5em)
      #body
      #v(0.5em)
    ]
  }
}

#let add_table(tb, caption: none) = {
  context {
    set figure(
      numbering: num => {
        let chap = counter(heading).get().first()
        numbering("1.1", chap, num)
      }
    )

    figure(
      tb,
      caption: caption
    )
  }
}

#let literature_survey(author, conjunctive, title, body) = {
  lit_survey_counter.step()

  strong(author)
  [ *[#context lit_survey_counter.display()]*]
  [
    #conjunctive _"#title."_

    #body
  ]
  v(1em)
}

#let header_footer_line() = {
  line(length: 100%, stroke: 3pt + rgb("#5F1E1E"))
  v(-0.85em)
  line(length: 100%, stroke: 0.75pt + rgb("#5F1E1E"))
}

#let footer_line() = {
  line(length: 100%, stroke: 0.75pt + rgb("#5F1E1E"))
  v(-0.85em)
  line(length: 100%, stroke: 3pt + rgb("#5F1E1E"))
}

// --- Front-matter helpers ---

// Centered line of text with the given size/weight/fill.
#let front_center_line(size, body, weight: "regular", fill: black) = {
  align(center)[
    #set text(size: size, weight: weight, fill: fill)
    #body
  ]
}

// A certificate signature column: label above a rule above the name.
#let signature_cell(label, name) = {
  align(center, stack(dir: ltr)[
    #text(size: 13pt)[#label]
    #v(4pt)
    #line(length: 60%, stroke: 0.5pt)
    #v(4pt)
    #text(size: 13pt, weight: "bold")[#name]
  ])
}

// A row of signature columns for the certificate page.
#let signature_block(entries) = {
  grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 2em,
    ..entries.map(e => signature_cell(e.at(0), e.at(1))),
  )
}
