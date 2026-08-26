#!/usr/bin/env bash

set -e

ros_setup_path="${F1_ROS_SETUP_PATH:-/opt/ros/jazzy/setup.bash}"
if [ ! -r "$ros_setup_path" ]; then
    echo "F1 container is missing ROS setup: $ros_setup_path" >&2
    exit 1
fi
venv_path="${F1_VENV_PATH:-/opt/rlinf-venv}"
if [ ! -r "$venv_path/bin/activate" ]; then
    echo "F1 container is missing Python venv: $venv_path" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ros_setup_path"
# shellcheck disable=SC1090
source "$venv_path/bin/activate"
exec "$@"
