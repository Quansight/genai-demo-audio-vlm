FROM mambaorg/micromamba

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml .
RUN micromamba install --yes -n base --file ./environment.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
