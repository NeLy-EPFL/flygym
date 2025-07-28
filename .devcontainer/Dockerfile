FROM python:3.13-bookworm

ARG USERNAME=flygym
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt update && \
    apt-get install -y libegl1-mesa-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set renderer
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
USER $USERNAME