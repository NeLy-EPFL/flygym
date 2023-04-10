#!/user/bin/env bash
set -e
set -u

# Find the absolute path of the flygym folder
docker_dir=$(cd $(dirname $0); pwd -P)
proj_root=$(dirname $(dirname $docker_dir))
echo "Mounting flygym folder as a volume: $proj_root > /home/gymuser/flygym"

# Make a new directory for sharing other data between the container and the
# host
data_path="$HOME/share/flygym_ig_container_data"
echo "Mounting additional data folder as volume $data_path > /home/gymuser/data"
mkdir -p $data_path

if [ $# -eq 0 ]
then
    echo "Starting container without display"
    docker run \
        -it \
        --network=host \
        --gpus=all \
        --name=flygym_ig_container \
        --ipc=host \
        -v "$proj_root:/home/gymuser/flygym" \
        -v "$data_path:/home/gymuser/data" \
        flygym_ig \
        /bin/bash
else
    echo "Starting container with display: '$1'"
    xhost +
    # IPC added to make the container use the host's shared memory
    docker run \
        -it \
        --network=host \
        --gpus=all \
        --name=flygym_ig_container \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=$1 \
        --ipc=host \
        -v "$proj_root:/home/gymuser/flygym" \
        -v "$data_path:/home/gymuser/data" \
        flygym_ig \
        /bin/bash
    xhost -
fi
