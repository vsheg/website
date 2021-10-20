FROM python:3.10-slim
LABEL maintainer="Vsevolod Shegolev <v@sheg.cc>"

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    FLASK_APP=app

RUN pip install poetry gunicorn

WORKDIR /app
COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . ./
WORKDIR /app/website

EXPOSE 8000
CMD gunicorn --workers=2 --bind 0.0.0.0:8000 --reload app:app
