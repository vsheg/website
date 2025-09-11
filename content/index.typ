Hello World

#let post-tile(link, title, description) = {
  html.elem("div", attrs: (class: "post-tile"), {
    html.elem("a", attrs: (href: link), {
      html.elem("h2", title)
      html.elem("p", description)
    })
  })
}

#post-tile(
  "posts/2018-11-05-pH-titration-mathematica",
  "This is a title",
  "Description of the post",
)
