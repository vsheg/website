#import "@preview/marginalia:0.2.3" as marginalia: note as margin
#import "@preview/physica:0.9.5": *


// Template
#let post(date: none, date-modified: none, categories: (), references: none, doc) = {
  // Marginalia setup
  // set page(width: 600pt)
  // show: marginalia.setup.with(outer: (width: 150pt))

  // Code style (inline)
  //
  show raw.line: box.with(fill: gray.transparentize(80%), outset: 0.3em, radius: 0.4em)

  set math.equation(numbering: "(1)")
  show math.equation.where(block: false): it => html.elem("span", html.frame(it))
  show math.equation.where(block: true): html.frame


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


// Note block
#let note(content) = box(fill: gray.transparentize(90%), outset: 0.5em, inset: 0.5em, content)


// Code style (blocks)
#import "@preview/zebraw:0.5.5": *
#let code(content, path: none) = {
  if path != none {
    let path = raw(path)
  }

  zebraw(
    header: path,
    content,
  )
}


// Math notation
#let const = "const"
