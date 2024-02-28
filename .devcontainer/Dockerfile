FROM mcr.microsoft.com/vscode/devcontainers/base:1-ubuntu-22.04
RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        ripgrep \
        tig && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
