FROM python:3.11-slim-bookworm

WORKDIR /app

COPY audiovlm_demo ./audiovlm_demo
COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .

# TODO: Fix this.
# See https://setuptools-scm.readthedocs.io/en/stable/usage/#with-dockerpodman
ARG PSEUDO_VERSION=1
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AUDIOVLM_DEMO=${PSEUDO_VERSION} pip install --no-cache-dir .

EXPOSE 5006

CMD panel serve audiovlm_demo/main.py --address 0.0.0.0 --allow-websocket-origin=127.0.0.1:5006
