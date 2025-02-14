FROM mambaorg/micromamba

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml .
RUN micromamba install --yes -n base --file ./environment.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1


COPY --chown=$MAMBA_USER:$MAMBA_USER audiovlm_demo ./audiovlm_demo
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml .
COPY --chown=$MAMBA_USER:$MAMBA_USER README.md .

RUN pip install --no-cache-dir .

EXPOSE 5006

CMD panel serve --dev audiovlm_demo/main.py --address 127.0.0.1 --allow-websocket-origin=127.0.0.1:5006
