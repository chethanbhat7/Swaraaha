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
      caption: caption,
    )
  }
}

#let add_image(img, caption: none) = {
  context {
    set figure(
      numbering: num => {
        let chap = counter(heading).get().first()
        numbering("1.1", chap, num)
      }
    )

    figure(
      img,
      caption: caption,
    )
  }
}

#let literature_survey(author, conjunctive, title, body) = {
  lit_survey_counter.step()

  author
  [ [#context lit_survey_counter.display()]]
  [
    #conjunctive #body
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

// A certificate signature column: a rule above the label above the name.
#let signature_cell(label, name, line_gap: 4pt, name_gap: 4pt, label_leading: 0.65em, line_length: 60%) = {
  align(center, stack(dir: ltr)[
    #line(length: line_length, stroke: 0.5pt)
    #v(line_gap)
    #par(leading: label_leading)[#text(size: 13pt)[#label]]
    #v(name_gap)
    #text(size: 13pt, weight: "bold")[#name]
  ])
}

// A row of signature columns for the certificate page.
// Entries are `(label, name)` or `(label, name, name_gap, label_leading, line_length)`.
#let signature_block(entries, columns: (1fr, 1fr, 1fr), gutter: 2em, line_gap: 4pt, name_gap: 4pt, label_leading: 0.65em, line_length: 60%) = {
  grid(
    columns: columns,
    column-gutter: gutter,
    ..entries.map(e => {
      let ng = if e.len() > 2 { e.at(2) } else { name_gap }
      let ll = if e.len() > 3 { e.at(3) } else { label_leading }
      let ln = if e.len() > 4 { e.at(4) } else { line_length }
      signature_cell(e.at(0), e.at(1), line_gap: line_gap, name_gap: ng, label_leading: ll, line_length: ln)
    }),
  )
}
