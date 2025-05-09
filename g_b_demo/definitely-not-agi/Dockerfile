FROM ubuntu:25.04@sha256:79efa276fdefa2ee3911db29b0608f8c0561c347ec3f4d4139980d43b168d991

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install TeXLive and other required packages
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  texlive \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-fonts-extra \
  texlive-science \
  zip \
  latexmk \
  make \
  git \
  ca-certificates \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Configure Git SSL settings
RUN git config --global http.sslCAinfo || echo "No CA info set." && \
  update-ca-certificates && \
  git config --global http.sslCAinfo /etc/ssl/certs/ca-certificates.crt

# Set working directory
WORKDIR /github/workspace
