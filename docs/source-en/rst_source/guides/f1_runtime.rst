F1 Two-Node Runtime
===================

Build the F1 runtime as a normal two-node RLinf deployment. The GPU server owns
training. The robot-side ARM64 container owns ROS 2, the Controller, and the Env
Worker. The reference deployment is validated on NVIDIA Thor. Both nodes join
Ray after their rank and network interface are set.

Runtime Boundary
----------------

Keep training and robot communication on separate nodes. Run the RLinf entry
script only on the GPU head.

.. list-table::
   :header-rows: 1
   :widths: 20 22 22 36

   * - Node
     - Architecture
     - Runtime
     - Responsibility
   * - GPU server
     - Linux x86_64
     - Host ``uv`` / venv
     - Ray head, ActorGroup, RolloutGroup, optimizer, replay sampling, and logs.
   * - F1 robot node
     - Linux aarch64 / arm64
     - ROS 2 Jazzy container
     - Ray worker, EnvGroup, Controller ``backend: ros2``, sensor reads, and bounded robot commands.

.. warning::

   Set ``RLINF_NODE_RANK`` before ``ray start``. The setup script validates and
   exports the expected rank, but exporting it explicitly in your shell makes the
   node role visible before Ray captures the environment.

Prepare the GPU Environment
---------------------------

Install the embodied dependencies first. Then apply the F1 shared constraints
and Controller package to the same environment:

.. code-block:: bash

   bash requirements/install.sh embodied \
     --env dummy \
     --python 3.12.3 \
     --venv .venv-f1 \
     --install-rlinf
   F1_ROBOT_CONTROLLER_PACKAGE=<wheel-path-package-url-or-pinned-git-url> \
     bash requirements/install.sh f1-realworld \
       --python 3.12.3 \
       --venv .venv-f1
   source .venv-f1/bin/activate
   python -c "import f1_robot_controller, platform, ray; assert platform.python_version() == '3.12.3'; assert ray.__version__ == '2.57.0'; assert callable(f1_robot_controller.create_controller)"

What this does: the first command installs GPU-side training dependencies. The
second command pins the shared Ray runtime to ``2.57.0`` and installs the
Controller package needed by F1 configuration and environment imports. It does
not require ROS 2 on the GPU server.

Build the Robot-Side Image
--------------------------

Build the ARM64 image from the same RLinf source:

.. code-block:: bash

   export F1_ROBOT_CONTROLLER_PACKAGE_URL=<package-url-or-pinned-git-url>
   docker build --platform linux/arm64 \
     -f docker/f1/Dockerfile \
     --build-arg F1_ROBOT_CONTROLLER_PACKAGE_URL="${F1_ROBOT_CONTROLLER_PACKAGE_URL}" \
     -t f1-rlinf-env:runtime-simplified .

What this does: the Dockerfile creates a Python ``3.12.3`` venv, installs the
shared F1 constraints, installs the supplied Controller package with
``--no-deps``, copies RLinf source, runs ``requirements/install.sh f1-realworld``,
and imports ROS message packages, Ray, RLinf, Controller, Pillow, ImageIO, and
Torch during the build. For Docker, ``F1_ROBOT_CONTROLLER_PACKAGE_URL`` must be
a package URL or pinned git URL that the build can fetch; the Dockerfile does
not copy a host-local wheel into the image. For local
``requirements/install.sh f1-realworld`` runs, ``F1_ROBOT_CONTROLLER_PACKAGE``
may point to a local wheel path, package URL, or pinned git URL. These are
distribution inputs, not runtime approval tokens or safety artifacts.

Use the default entrypoint. It sources ``/opt/ros/jazzy/setup.bash``, activates
``/opt/rlinf-venv``, then executes your command.

Edit the Runtime YAML
---------------------

Use one YAML as the runtime contract. The real robot config is:
``examples/embodiment/config/realworld_f1_peg_rlpd_cnn_async.yaml``.

Keep these fields inline:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor: {node_group: gpu, placement: 0}
       rollout: {node_group: gpu, placement: 0}
       env: {node_group: f1, placement: 0}
   env:
     train:
       max_episode_steps: 10
       max_steps_per_rollout_epoch: 10
       override_cfg:
         controller:
           backend: ros2
         action_scale:
           tcp_position_m: 0.005
           tcp_orientation_deg: 1.0
           gripper_percent_closed: 10.0
         motion_envelope:
           gripper_percent_closed: [0.0, 100.0]
         max_num_steps: 10

What this does: ``controller.backend`` selects the Controller implementation;
``action_scale`` converts normalized policy actions to physical deltas;
``motion_envelope`` bounds the Controller command. Override model and log paths
with normal Hydra keys such as ``actor.model.model_path=/models/f1-cnn`` and
``runner.logger.log_path=/runs/f1-peg``.

The three step fields accept positive integers. Keep them aligned for a single
fixed-length collection window, or set them independently when the episode and
rollout boundaries intentionally differ.

Run Controller Diagnostics
--------------------------

Run these commands on the robot-side node before a robot run:

.. code-block:: bash

   controller_json="$(mktemp /tmp/f1-controller.XXXXXX.json)"
   EMBODIED_PATH="$PWD/examples/embodiment" \
   python - "$controller_json" <<'PY'
   import json
   import sys
   from pathlib import Path

   from hydra import compose, initialize_config_dir
   from omegaconf import OmegaConf

   output = Path(sys.argv[1])
   with initialize_config_dir(
       config_dir=str(
           Path.cwd() / "examples" / "embodiment" / "config"
       ),
       version_base=None,
   ):
       cfg = compose(config_name="realworld_f1_peg_rlpd_cnn_async")
   override_cfg = OmegaConf.to_container(
       cfg.env.train.override_cfg,
       resolve=True,
   )
   controller = dict(override_cfg["controller"])
   controller["motion_envelope"] = override_cfg["motion_envelope"]
   output.write_text(json.dumps(controller, indent=2, sort_keys=True) + "\n")
   PY
   f1-controller doctor --config "$controller_json"
   f1-controller read-state --config "$controller_json"

What this does: you still edit one Hydra YAML, but the Controller CLI receives
the temporary controller JSON it owns: the extracted ``controller`` mapping plus
the sibling ``motion_envelope``. ``f1-controller doctor`` checks ROS 2
discovery, required topics, message types, observation freshness, observation
skew, and read-only connectivity. ``f1-controller read-state`` prints the
normalized current robot state.

Start the Ray Cluster
---------------------

Start the GPU head first:

.. code-block:: bash

   export F1_RUNTIME_ROLE=gpu
   export F1_VENV_PATH="$PWD/.venv-f1"
   export RLINF_COMM_NET_DEVICES=<GPU_INTERFACE>
   export RLINF_NODE_RANK=0
   source ray_utils/realworld/f1/setup_before_ray.sh
   ray start --head --port=6379 --node-ip-address=<GPU_SERVER_IP>

Start the robot-side worker in the container:

.. code-block:: bash

   docker run --rm --network host --name f1-ray-worker f1-rlinf-env:runtime-simplified bash -lc '
     export F1_RUNTIME_ROLE=thor
     export F1_VENV_PATH=/opt/rlinf-venv
     export RLINF_COMM_NET_DEVICES=<THOR_INTERFACE>
     export RLINF_NODE_RANK=1
     source ray_utils/realworld/f1/setup_before_ray.sh
     ray start --address=<GPU_SERVER_IP>:6379 --node-ip-address=<THOR_IP> --block
   '

What this does: the setup script validates Python ``3.12.3``, Ray ``2.57.0``,
the selected role, and the communication interface. On the robot-side node it
also sources ROS 2 and sets ``RMW_IMPLEMENTATION=rmw_cyclonedds_cpp``. The
``thor`` value is the current runtime role token for the ARM64 robot node.

Prepare RLPD Data
-----------------

The F1 RLPD configuration requires a non-empty demonstration buffer. Set
``algorithm.demo_buffer.load_path`` to an RLinf replay buffer collected for the
same task, with the same observation and 14-dimensional action shapes.

If you do not have demonstrations, run online SAC by removing the demo buffer:

.. code-block:: bash

   '~algorithm.demo_buffer'

Run Training
------------

Confirm the robot workspace is clear and a trained operator is supervising the
run. Launch training from the GPU head:

.. code-block:: bash

   EMBODIED_PATH="$PWD/examples/embodiment" \
   python examples/embodiment/train_async.py \
     --config-name realworld_f1_peg_rlpd_cnn_async \
     actor.model.model_path=/path/to/RLinf-ResNet10-pretrained \
     algorithm.demo_buffer.load_path=/path/to/f1-peg-demo-buffer

For online SAC, append ``'~algorithm.demo_buffer'``. RLinf collects online
transitions, performs SAC updates, and syncs Actor weights back to Rollout.

Audit Logs, Replay, and Stop Safely
-----------------------------------

Check the run before you repeat it.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Item
     - Command or Check
   * - Ray cluster
     - ``ray status`` on the GPU head should show two live nodes before real runs.
   * - Training logs
     - Open TensorBoard under ``runner.logger.log_path`` and inspect replay counts, SAC losses, and timing metrics.
   * - Online replay
     - Confirm that the replay buffer under ``runner.logger.log_path`` is written and its sample count grows.
   * - Safety stop
     - Stop the runner with ``Ctrl+C``. Run ``ray stop`` on the GPU server and stop the robot-side container. If Controller reports a fault, stop robot motion before collecting logs.

Troubleshooting
---------------

Use these checks before changing YAML:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Symptom
     - Check
   * - EnvGroup lands on the GPU node
     - Confirm ``RLINF_NODE_RANK=1`` was set before the robot node ran ``ray start``.
   * - Ray nodes cannot see each other
     - Confirm both nodes select interfaces on the same routable network and use host networking for the robot-side container.
   * - Controller diagnostics fail
     - Fix ROS 2 discovery, topic names, message types, or stale sensors before running RLinf.
   * - Replay does not grow
     - Confirm ``rollout.collect_transitions: true``, check EnvGroup logs, and fix Controller errors before another robot run.
