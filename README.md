# Hello,

this repo contains the code of my personal website, which I plan to use to deploy some ML models in the future.

# Tech stack

## Python

- [Flask] micro web framework
- [Gunicorn] WSGI server for production
- [Poetry] as a dependency manager
- [pre-commit], [Black], [Pylint] for automated code review

## Web

- [Bulma] CSS framework
- [Buefy] web framework, based on [Bulma] and [Vue.js], that provides useful JavaScript facilities
- [Pug] HTML preprocessor, implemented in Python ([PyPugJS])
- ~~[SASS] CSS preprocessor~~

## Server

- [Nginx] as a proxy
- [Let's Encrypt] as a provider for HTTPS

## CI/CD

- [Docker] & [Docker Compose] for containerization
- ~~[GitHub Actions] for testing and deploy to my VPS~~

[Flask]: https://github.com/pallets/flask
[Gunicorn]: https://github.com/benoitc/gunicorn
[Poetry]: https://github.com/python-poetry/poetry
[pre-commit]: https://pre-commit.com
[Black]: https://github.com/psf/black
[Pylint]: https://github.com/PyCQA/pylint
[Bulma]: https://github.com/jgthms/bulma
[Buefy]: https://github.com/buefy/buefy
[Vue.js]: https://github.com/vuejs/vue
[Pug]: https://pugjs.org
[PyPugJS]: https://github.com/kakulukia/pypugjs
[SASS]: https://github.com/sass/sass
[Nginx]: https://www.nginx.com
[Let's Encrypt]: https://letsencrypt.org
[Docker]: https://www.docker.com
[Docker Compose]: https://github.com/docker/compose
[GitHub Actions]: https://docs.github.com/en/actions
