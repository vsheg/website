#import "../_template.typ": web
#import "@preview/fontawesome:0.6.0": fa-icon

#let contact-item(icon, text, href) = context {
  let icon = fa-icon(icon)

  if target() == "html" {
    let icon = html.frame(icon)
  }

  icon + [ ] + link(href, raw(text))
}


#let content = [
  #image("profile.jpg")

  Vsevolod Shegolev

  a charming guy interested in computational biochemistry, machine learning, and _in silico_ drug design

  - PhD student in *Biotechnology* \@ Lomonosov MSU
  - MSc in *Chemical Enzymology* \@ Lomonosov MSU
  - BSc in *Computational Chemistry* \@ Lomonosov MSU


  #contact-item(
    "envelope",
    "mail@vsheg.com",
    "mailto:mail@vsheg.com?&body=Hi%20Vsevolod%2C%0A%0A%0A",
  )

  #contact-item(
    "send",
    "t.me/vsheg",
    "https://t.me/vsheg",
  )

  #contact-item(
    "github",
    "@vsheg",
    "https://github.com/vsheg",
  )

  #contact-item(
    "linkedin",
    "in/vsheg",
    "https://www.linkedin.com/in/vsheg/",
  )
]

#context {
  if target() == "html" {
    web(content)
  } else {
    content
  }
}
