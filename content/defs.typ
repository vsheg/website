#import "@preview/physica:0.9.5": *


#let margin(content) = context {
  if target() == "html" {
    html.elem("span", attrs: (class: "margin-body"), content)
  }
}

// Template
#let post(date: none, date-modified: none, categories: (), references: none, doc) = {
  show raw.line: box.with(fill: gray.transparentize(80%), outset: 0.3em, radius: 0.4em)

  set math.equation(numbering: "(1)")

  show math.equation.where(block: false): it => {
    if target() == "html" {
      html.elem("span", attrs: (role: "math"), html.frame(it))
    } else {
      it
    }
  }

  show math.equation.where(block: true): it => {
    if target() == "html" {
      html.elem("figure", attrs: (role: "math"), html.frame(it))
    } else {
      it
    }
  }


  set par(justify: true, linebreaks: "optimized")
  set text(hyphenate: true, costs: (hyphenation: 0%, runt: 50%, widow: 0%, orphan: 0%))

  show ref: it => {
    // Aliases
    let eq = math.equation
    let el = it.element

    // Equation reference
    if el != none and el.func() == eq {
      // Override equation references.
      return link(el.location(), numbering(
        el.numbering,
        ..counter(eq).at(el.location()),
      ))
    }

    if el != none and el.func() == heading { return smartquote() + it.element.body + smartquote() }

    it
  }

  show footnote: it => {
    if target() == "html" {
      html.elem("sup", attrs: (class: "footnote-ref"), it.numbering)
      html.elem("span", attrs: (class: "footnote-body"), super(it.numbering) + [ ] + it.body)
    }
  }

  date

  if date-modified != none {
    [ (updated: ] + date-modified + [) ]
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
