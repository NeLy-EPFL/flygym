FROM python:3.12-trixie

# Set shell to bash (instead of sh)
SHELL ["/bin/bash", "-c"]

# Change working directory
ENV HOME=/root
WORKDIR $HOME/flygym

# Install system dependencies
RUN apt update && \
    apt-get install -y libegl1-mesa-dev ffmpeg

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set renderer
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Copy flygym package to the container and install the package
ADD . $HOME/flygym/

# Install using uv
RUN uv sync --extra warp --extra examples --extra dev

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
