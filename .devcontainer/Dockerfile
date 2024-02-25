FROM mcr.microsoft.com/vscode/devcontainers/base:1-ubuntu-22.04
RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        file \
        git-lfs \
        inotify-tools \
        ripgrep \
        tig && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
# LaTeX
RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        build-essential \
        chktex \
        latexmk \
        texlive \
        texlive-extra-utils \
        texlive-fonts-extra \
        texlive-latex-extra \
        tig && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN curl -fsSL http://cpanmin.us | perl - App::cpanminus && \
    cpanm \
        File::HomeDir \
        Log::Dispatch::File \
        Log::Log4perl \
        Unicode::GCString \
        YAML::Tiny
# Quarto
RUN wget -c -O quarto.deb https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.550/quarto-1.4.550-linux-amd64.deb && \
    dpkg -i quarto.deb && \
    rm quarto.deb
