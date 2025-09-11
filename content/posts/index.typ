#import "../_template.typ": web

#show: web

#let post-tile(link, title, description) = {
  html.elem("div", attrs: (class: "post-tile"), {
    html.elem("a", attrs: (href: link), {
      html.elem("h2", title)
      html.elem("p", description)
    })
  })
}

= Posts

== 2025

#post-tile(
  "2025-04-12-derp-caddy",
  "Running Tailscale DERP server behind Caddy",
  "Run a self-hosted Tailscale DERP behind Caddy on 443 using TLS SNI and Docker.",
)

#post-tile(
  "2025-02-26-fedora-kernel-downgrade",
  "Fedora: Downgrade kernel",
  "Steps to downgrade Fedora kernel and select versions using grubby and DNF.",
)

#post-tile(
  "2025-01-22-xray-caddy",
  "Hiding VPN behind Caddy web server on port 443",
  "Hide Xray/VLESS behind Caddy with path-based routing and reverse proxy.",
)

== 2024

#post-tile(
  "2024-12-22-uv",
  "`uv` as a system-wide Python environment manager",
  "Manage a global Python environment with `uv` across OSes.",
)

== 2023

#post-tile(
  "2023-10-31-conditional-probability",
  "Visualizing conditional probability",
  "Derivation and visualization of conditional pdf for a bivariate Gaussian with distance measurement.",
)

== 2021

#post-tile(
  "2021-09-00-molecular-fingerprints",
  "Оценка структурной схожести молекул",
  "Методы поиска по подграфу, молекулярные отпечатки и метрики схожести.",
)

#post-tile(
  "2021-08-00-alchemy",
  "Вычислительная алхимия",
  "Алхимические трансформации, одинарная/двойная топология и интегрирование по пути.",
)

#post-tile(
  "2021-07-00-fep",
  "Теория возмущений свободной энергии",
  "FEP и TI: расчёт ΔF между состояниями и статистическое усреднение.",
)

#post-tile(
  "2021-06-00-docking",
  "Молекулярный докинг",
  "Семплинг поз и скоринг: подходы и алгоритмы докинга.",
)

#post-tile(
  "2021-05-00-metadynamics",
  "Метадинамика",
  "Коллективные переменные и смещающий потенциал для ускоренного семплинга.",
)

#post-tile(
  "2021-03-00-force-fields",
  "Силовые поля в молекулярной динамике",
  "Классические СП, типы атомов и разработка параметров.",
)

#post-tile(
  "2021-02-00-micro--and-macroparameters",
  "Оценка термодинамических микро- и макропараметров в классической молекулярной динамике",
  "Температура и давление в МД: термостаты и баростаты.",
)

#post-tile(
  "2021-01-00-molecular-interactions",
  "Молекулярные взаимодействия в классической молекулярной динамике",
  "Уравнение движения, потенциальная энергия и основные типы взаимодействий.",
)

== 2020

#post-tile(
  "2020-05-05-flammability-limits",
  "Modeling flammability limits in hydrogen-air mixtures using Mathematica",
  "Compute and visualize 3 flammability limits across P–T conditions.",
)

== 2018

#post-tile(
  "2018-11-05-pH-titration-mathematica",
  "Calculating titration curves in Mathematica",
  "Derive pH titration curve for a dibasic acid and locate equivalence points.",
)
