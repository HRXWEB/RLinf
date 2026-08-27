# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.15@sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0 AS uv

FROM ros:jazzy-ros-base@sha256:2589a8fba5257307857890173c069852c2abf913a0be7970f172478baecb09e4 AS f1-realworld-base

ARG TARGETARCH

RUN test "$TARGETARCH" = "arm64" && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        ros-jazzy-control-msgs \
        ros-jazzy-cv-bridge \
        ros-jazzy-diagnostic-msgs \
        ros-jazzy-geometry-msgs \
        ros-jazzy-map-msgs \
        ros-jazzy-nav-msgs \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-sensor-msgs \
        ros-jazzy-tf2-msgs && \
    apt-get clean && \
    find /var/lib/apt/lists -mindepth 1 -delete

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /opt/rlinf
COPY docker/f1/constraints/realworld-py312.txt /opt/rlinf/docker/f1/constraints/realworld-py312.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/rlinf-venv --python 3.12.3 --system-site-packages && \
    . /opt/rlinf-venv/bin/activate && \
    uv pip install -r /opt/rlinf/docker/f1/constraints/realworld-py312.txt

ARG F1_ROBOT_CONTROLLER_PACKAGE_URL

RUN --mount=type=cache,target=/root/.cache/uv \
    . /opt/rlinf-venv/bin/activate && \
    test -n "$F1_ROBOT_CONTROLLER_PACKAGE_URL" || \
    (echo "F1_ROBOT_CONTROLLER_PACKAGE_URL build arg is required; use a package URL or pinned git URL" >&2; exit 1) && \
    uv pip install --no-deps "$F1_ROBOT_CONTROLLER_PACKAGE_URL"

FROM f1-realworld-base AS runtime

ARG F1_ROBOT_CONTROLLER_PACKAGE_URL

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    F1_VENV_PATH=/opt/rlinf-venv \
    PATH=/opt/rlinf-venv/bin:${PATH} \
    PYTHONPATH=/opt/rlinf:${PYTHONPATH}

COPY pyproject.toml README.md LICENSE /opt/rlinf/
COPY rlinf /opt/rlinf/rlinf
COPY requirements/install.sh /opt/rlinf/requirements/install.sh
COPY --chmod=0555 docker/f1/entrypoint.sh /opt/rlinf/docker/f1/entrypoint.sh
COPY ray_utils/realworld/f1/setup_before_ray.sh /opt/rlinf/ray_utils/realworld/f1/setup_before_ray.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    . /opt/rlinf-venv/bin/activate && \
    F1_ROBOT_CONTROLLER_PACKAGE="$F1_ROBOT_CONTROLLER_PACKAGE_URL" \
    bash requirements/install.sh f1-realworld \
        --python 3.12.3 \
        --venv /opt/rlinf-venv \
        --use-mirror

RUN . /opt/ros/jazzy/setup.sh && \
    . /opt/rlinf-venv/bin/activate && \
    python -c "import PIL, control_msgs, cv_bridge, diagnostic_msgs, f1_robot_controller, geometry_msgs, imageio, map_msgs, nav_msgs, ray, rclpy, rlinf, sensor_msgs, tf2_msgs, torch; assert ray.__version__ == '2.57.0'; assert callable(f1_robot_controller.create_controller); assert callable(f1_robot_controller.load_controller_config); assert callable(f1_robot_controller.validate_motion_envelope)"

ENTRYPOINT ["/opt/rlinf/docker/f1/entrypoint.sh"]
CMD ["bash"]
