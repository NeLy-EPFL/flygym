# Follow Isaac Gym's isntruction to build the Isaac Gym image first
FROM isaacgym:latest

# Trick: change user and group IDs so that they're the same as the user/group
# IDs on the host machine. This way, when the user creates a file from
# *within* the container on a Docker volume, on the host *outside* the
# container, it will appear as if the user had created the file themselves.
ARG USER_ID
ARG GROUP_ID
USER root
RUN usermod -u $USER_ID gymuser
RUN groupmod -g $GROUP_ID gymuser
USER gymuser

# Copy FlyGym folder and install dependencies. The actual code will be updated
# upon `docker run`: the flygym project folder will be mounted as a volume.
WORKDIR /home/gymuser/flygym
COPY --chown=gymuser . .
RUN pip install -e ."[isaacgym]"
