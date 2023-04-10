#!/user/bin/env bash
set -e
set -u

# Find the absolute path of the flygym folder
docker_dir=$(cd $(dirname $0); pwd -P)
proj_root=$(dirname $(dirname $docker_dir))
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
