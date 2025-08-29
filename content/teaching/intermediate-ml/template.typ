#import "@preview/physica:0.9.5": *
#import "@preview/drafting:0.2.2": margin-note, set-page-properties, set-margin-note-defaults
#import "@preview/quick-maths:0.2.1": shorthands
#import "@preview/lilaq:0.2.0" as lq
#import "@preview/shadowed:0.2.0": shadowed
#import "defs.typ": *

//////////////
// TEMPLATE //
//////////////

#let draft-pattern = {
  let element = text(size: 2em, fill: gray.opacify(-90%))[*DRAFT*]
  let pattern = tiling(size: (90pt, 40pt), element)
  rotate(-25deg, rect(width: 150%, height: 150%, fill: pattern))
}

#let template(is-draft: true, doc) = {
  // page setup
  let content-width = 180mm
  let margin-size = 10mm
  let content-heigh = 250mm
  let full-width = content-width + 2 * margin-size
  let full-heigh = content-heigh + 2 * margin-size

  set page(
    width: full-width,
    height: full-heigh,
    margin: (y: margin-size, left: margin-size, right: 0.34 * content-width),
    background: if is-draft { draft-pattern } else { none },
  )

  // important to margin notes from drafting package
  set-page-properties()

  // font
  let font-size = 9pt
  set text(size: font-size, hyphenate: true, font: "New Computer Modern", costs: (hyphenation: 10%))
  show raw: set text(font: "Menlo", size: 0.9em)

  // math equations
  set math.equation(numbering: "(1)")

  show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      // Override equation references.
      link(el.location(), numbering(el.numbering, ..counter(eq).at(el.location())))
    } else {
      // Other references as usual.
      it
    }
  }

  // text
  show "i.e.": set text(style: "italic")
  show "e.g.": set text(style: "italic")
  show "etc.": set text(style: "italic")
  show "cf.": set text(style: "italic")
  show "vs.": set text(style: "italic")
  set par(justify: true)

  // lists
  set list(
    marker: (
      text(font: "Menlo", size: 1.5em, baseline: -0.2em, "✴", fill: accent-color),
      text(size: 0.6em, baseline: +0.2em, "➤", fill: ghost-color),
    ),
  )

  // headings
  show heading: set text(fill: accent-color)
  set heading(numbering: "1")
  let heading_counter = counter("heading_counter")

  show heading.where(level: 1): it => {
    heading_counter.step()
    if heading_counter.get().at(0) > 0 {
      pagebreak(weak: true)
    }
    set text(size: font-size * 1.1)
    set block(below: 1em)
    it
  }

  show heading.where(level: 2): it => {
    set text(size: font-size * 0.9)
    v(1em)
    text(it.body) + [: ]
  }

  // quick math shorthands
  show: shorthands.with(..replacements)

  // tables
  set table(stroke: none)
  set table(align: center + horizon)
  set table(
    fill: (_, y) => if (y == 0) { accent-color.transparentize(80%) } else {
      if calc.even(y) { ghost-color.transparentize(80%) }
    },
  )

  // grids
  set grid(column-gutter: 1em)

  // references
  show bibliography: none

  // begin document
  doc

  bibliography(title: [References], style: "apa", "assets/citations.bib")
}

///////////////////////
// STYLING FUNCTIONS //
///////////////////////

// Math annotation
#set math.cancel(stroke: black.transparentize(50%))

#let note-style(content) = {
  set math.equation(numbering: none)
  show math.equation.where(block: true): set block(spacing: 0.5em)
  set text(size: 0.9em, fill: luma(20%))
  content
}

#let margin(title: none, ..content) = {
  show: note-style

  if content.pos().len() == 1 {
    content = content.at(0)
  } else {
    title = content.at(0)
    content = content.pos().slice(1).join(linebreak())
  }

  if title != none {
    title = strong(title) + [: ]
  }

  v(0pt, weak: true)
  margin-note(side: right, stroke: none, title + content)
}

#let uplift(content) = {
  pad(
    x: -2mm,
    y: -1mm,
    shadowed(
      radius: 2mm,
      inset: 2mm,
      shadow: 2mm,
      fill: gradient.linear(luma(99%), luma(98%), angle: 30deg),
      content,
    ),
  )
}

#let note(cols: 1, title: none, content) = {
  show: note-style

  title = if title != none {
    text(strong(title) + [: ], size: 0.9em)
  }

  set align(center)

  uplift({
    set align(left)
    show: it => columns(cols, it)
    title
    content
  })
}

#let example(cols: 1, content) = {
  note(cols: cols, title: [Example], content)
}

#let divider = {
  line(length: 100%, stroke: (paint: ghost-color, thickness: 0.5pt, dash: "loosely-dashed"))
}

// MATH ANNOTATION

#let focus(content) = {
  text(fill: accent-color, content)
}

#let comment(content) = text(fill: ghost-color, size: 0.8em, $&$ + content)

// Default color palette
#import "@preview/typpuccino:0.1.0": latte
#let palette = latte
#let colors = (palette.teal, palette.pink, palette.flamingo, palette.mauve, palette.green)

// Multifigure
#import "@preview/subpar:0.2.1": grid as multi-figure
#let multi-figure = multi-figure.with(numbering-sub: "1a:")
