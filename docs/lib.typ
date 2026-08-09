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
