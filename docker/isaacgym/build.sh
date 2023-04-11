#!/user/bin/env bash
set -e
set -u

# Find the absolute path of the flygym folder
ig_docker_dir=$(realpath -P $(dirname $0))
proj_root=$(dirname $(dirname $ig_docker_dir))
cd $proj_root
echo "Working in $(pwd)"

# See Dockerfile for explaination of user and group IDs
docker build \
    --network host \
    -t flygym_ig \
    -f docker/isaacgym/Dockerfile \
    --build-arg USER_ID=$(id -u $(whoami)) \
    --build-arg GROUP_ID=$(id -g $(whoami)) \
    .
