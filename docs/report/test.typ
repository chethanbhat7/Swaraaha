#let margins = (
  left: 1.25in,
  top: 0.75in,
  right: 1in,
  bottom: 0.75in,
)

#let inch(length) = calc.round(length / 1in, digits: 2)

#set page(
  margin: margins,
  header: rect(width: 100%, fill: aqua)[
    Header — Top margin: #inch(margins.top) in
  ],
  footer: rect(width: 100%, fill: aqua)[
    Footer — Bottom margin: #inch(margins.bottom) in
  ],
  number-align: center,
)

#rect(
  width: 100%,
  height: 100%,
  fill: aqua.lighten(40%),
)[
  Left margin: #inch(margins.left) in \
  Right margin: #inch(margins.right) in
]
