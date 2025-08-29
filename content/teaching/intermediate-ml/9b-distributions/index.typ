#import "../defs.typ": *

#let dist-plot(bins, x-label: $x$, y-label: $p(x)$) = discrete-plot(x-label: x-label, y-label: y-label, ys: bins)

#let plot1 = dist-plot((0, 0, 0, 1, 0, 0, 0))
#let plot2 = dist-plot((0.01, 0.05, 0.24, 0.4, 0.24, 0.05, 0.01))
#let plot3 = dist-plot((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))

#subpar.grid(
  figure(plot1, caption: [$H(p) = 1 dot log 1 = 0$]),
  figure(plot2, caption: [$0 <= H(p) <= log 7$]),
  figure(plot3, caption: [$H(p) = 7 dot 1/7 dot log 7 = log 7$]),
  columns: (1fr, 1fr, 1fr),
  caption: [Entropies of different discrete distributions.],
)

