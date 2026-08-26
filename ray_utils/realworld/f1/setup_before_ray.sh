#!/bin/bash

F1_RUNTIME_PYTHON_VERSION="3.12.3"
F1_RUNTIME_RAY_VERSION="2.57.0"

_f1_setup_script_path() {
    if [ -n "${BASH_VERSION:-}" ]; then
        printf '%s\n' "${BASH_SOURCE[0]}"
    elif [ -n "${ZSH_VERSION:-}" ]; then
        printf '%s\n' "${(%):-%x}"
    else
        echo "Source this setup script from Bash or Zsh so Ray inherits the validated environment." >&2
        return 1
    fi
}

_f1_setup_is_sourced() {
    if [ -n "${BASH_VERSION:-}" ]; then
        [ "${BASH_SOURCE[0]}" != "$0" ]
    elif [ -n "${ZSH_VERSION:-}" ]; then
        case "$ZSH_EVAL_CONTEXT" in
            *:file:*) return 0 ;;
            *) return 1 ;;
        esac
    else
        return 1
    fi
}

_f1_source_ros_setup() {
    local ros_setup_path source_status had_nounset
    ros_setup_path="${F1_ROS_SETUP_PATH:-/opt/ros/jazzy/setup.bash}"
    if [ ! -f "$ros_setup_path" ]; then
        echo "F1 Thor runtime is missing ROS setup: $ros_setup_path" >&2
        return 1
    fi

    case "$-" in
        *u*)
            had_nounset=1
            set +u
            ;;
        *)
            had_nounset=0
            ;;
    esac
    # shellcheck disable=SC1090
    source "$ros_setup_path"
    source_status=$?
    if [ "$had_nounset" = "1" ]; then
        set -u
    fi
    return "$source_status"
}

if ! _f1_setup_is_sourced; then
    echo "Source this Bash or Zsh script so Ray inherits the validated environment." >&2
    exit 1
fi

_f1_setup_before_ray() {
    local script_path script_dir repo_path venv_path expected_rank actual_python actual_ray
    if ! script_path="$(_f1_setup_script_path)"; then
        return 1
    fi
    script_dir="$(cd "$(dirname "$script_path")" && pwd)"
    repo_path="$(cd "$script_dir/../../.." && pwd)"
    venv_path="${F1_VENV_PATH:-/opt/rlinf-venv}"

    if [ -z "${F1_RUNTIME_ROLE:-}" ]; then
        echo "Set F1_RUNTIME_ROLE=gpu|thor before sourcing this script." >&2
        return 1
    fi
    if [ -z "${RLINF_COMM_NET_DEVICES:-}" ]; then
        echo "Set RLINF_COMM_NET_DEVICES to the interface shared by both Ray nodes." >&2
        return 1
    fi
    if [ ! -f "$venv_path/bin/activate" ] || [ ! -x "$venv_path/bin/python" ]; then
        echo "F1 venv is missing activate or Python executable: $venv_path" >&2
        return 1
    fi
    case "$F1_RUNTIME_ROLE" in
        gpu) expected_rank="0" ;;
        thor) expected_rank="1" ;;
        *)
            echo "F1_RUNTIME_ROLE must be gpu or thor (got $F1_RUNTIME_ROLE)." >&2
            return 1
            ;;
    esac
    if [ -n "${RLINF_NODE_RANK:-}" ] && [ "$RLINF_NODE_RANK" != "$expected_rank" ]; then
        echo "F1 $F1_RUNTIME_ROLE runtime requires RLINF_NODE_RANK=$expected_rank (got $RLINF_NODE_RANK)." >&2
        return 1
    fi

    actual_python="$("$venv_path/bin/python" -c 'import platform; print(platform.python_version())')"
    if [ "$actual_python" != "$F1_RUNTIME_PYTHON_VERSION" ]; then
        echo "F1 Python version mismatch: expected $F1_RUNTIME_PYTHON_VERSION, got $actual_python." >&2
        return 1
    fi
    actual_ray="$("$venv_path/bin/python" -c 'import ray; print(ray.__version__)')"
    if [ "$actual_ray" != "$F1_RUNTIME_RAY_VERSION" ]; then
        echo "F1 Ray version mismatch: expected $F1_RUNTIME_RAY_VERSION, got $actual_ray." >&2
        return 1
    fi

    # Validation is complete. Only now mutate variables that Ray captures.
    export F1_VENV_PATH="$venv_path"
    # shellcheck disable=SC1090
    source "$venv_path/bin/activate"
    if [ "$F1_RUNTIME_ROLE" = "thor" ]; then
        if ! _f1_source_ros_setup; then
            return 1
        fi
        export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
    fi
    export PYTHONPATH="$repo_path:${PYTHONPATH:-}"
    export RLINF_COMM_NET_DEVICES
    export RLINF_NODE_RANK="$expected_rank"
    printf 'F1 runtime ready: role=%s python=%s ray=%s rank=%s\n' \
        "$F1_RUNTIME_ROLE" "$F1_RUNTIME_PYTHON_VERSION" \
        "$F1_RUNTIME_RAY_VERSION" "$RLINF_NODE_RANK"
}

_f1_setup_before_ray
_f1_setup_status=$?
unset -f _f1_source_ros_setup
unset -f _f1_setup_is_sourced
unset -f _f1_setup_script_path
unset -f _f1_setup_before_ray
return "$_f1_setup_status"
