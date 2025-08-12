#let const = "const"
#let dd(arg) = math.serif("d") + arg

#let post(date: none, date-modified: none, categories: (), doc) = {
  set math.equation(numbering: "(1)")

  doc
}
