#import "@preview/physica:0.9.5": *

#let const = "const"
#let dd(arg) = math.serif("d") + arg

#import "@preview/marge:0.1.0": sidenote

#let margin = sidenote.with(padding: 1em)

#let post(date: none, date-modified: none, categories: (), references: none, doc) = {
  set math.equation(numbering: "(1)")

  set page(margin: (right: 7cm))
  set par(justify: true, linebreaks: "optimized")
  set text(hyphenate: true, costs: (hyphenation: 0%, runt: 50%, widow: 0%, orphan: 0%))

  set heading(numbering: "1.")

  show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      numbering(
        el.numbering,
        ..counter(eq).at(el.location()),
      )
    } else {
      it
    }
  }

  doc

  if references != none {
    bibliography(references)
  }
}

#let const = "const"
