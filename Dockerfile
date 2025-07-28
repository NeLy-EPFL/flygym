FROM python:3.13-bookworm

# Set shell to bash (instead of sh)
SHELL ["/bin/bash", "-c"]

# Change working directory
ENV HOME=/root
WORKDIR $HOME/flygym

# Install system dependencies
RUN apt update && \
    apt-get install -y libegl1-mesa-dev ffmpeg

# Set renderer
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Copy flygym package to the container and install the package
ADD . $HOME/flygym/

# Set up virtual environment and install dependencies
RUN pip install -e ".[examples,dev]"

# Set entrypoint
ENTRYPOINT ["/bin/bash"]