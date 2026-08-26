F1 双节点运行时
===============

把 F1 runtime 当作普通双节点 RLinf 部署来构建。GPU server 负责训练；Thor
container 负责 ROS 2、Controller ``0.2.0`` 和 Env Worker。两个节点都要先设置
rank 和网络接口，再加入 Ray。

Runtime Boundary
----------------

把训练和机器人通信放在不同节点。只在 GPU head 上运行 RLinf entry script。

.. list-table::
   :header-rows: 1
   :widths: 20 22 22 36

   * - 节点
     - 架构
     - Runtime
     - 责任
   * - GPU server
     - Linux x86_64
     - Host ``uv`` / venv
     - Ray head、ActorGroup、RolloutGroup、optimizer、replay sampling 和 logs。
   * - F1 Thor
     - Linux aarch64 / arm64
     - ROS 2 Jazzy container
     - Ray worker、EnvGroup、Controller ``backend: ros2``、sensor reads 和 bounded robot commands。

.. warning::

   在 ``ray start`` 之前设置 ``RLINF_NODE_RANK``。setup script 会验证并导出
   预期 rank，但在 shell 中显式 export 可以让 Ray 捕获环境前就看清节点角色。

Prepare the GPU Environment
---------------------------

在 GPU server 上安装训练环境：

.. code-block:: bash

   bash requirements/install.sh embodied \
     --env dummy \
     --python 3.12.3 \
     --venv .venv-f1 \
     --install-rlinf
   source .venv-f1/bin/activate
   python -c "import platform, ray; assert platform.python_version() == '3.12.3'; assert ray.__version__ == '2.57.0'"

What this does: 安装 RLinf 和 GPU 侧 training dependencies。GPU server 不需要
ROS 2 imports。

Build the Thor Image
--------------------

从同一份 RLinf source 构建 ARM64 image：

.. code-block:: bash

   export F1_ROBOT_CONTROLLER_PACKAGE_URL=git+ssh://git.example.com/f1-robot-controller.git@<controller-0.2.0-commit>
   docker build --platform linux/arm64 \
     -f docker/f1/Dockerfile \
     --build-arg F1_ROBOT_CONTROLLER_PACKAGE_URL="${F1_ROBOT_CONTROLLER_PACKAGE_URL}" \
     -t f1-rlinf-env:runtime-simplified .

What this does: Dockerfile 创建 Python ``3.12.3`` venv，安装共享 F1
constraints，用 ``--no-deps`` 安装用户提供的 Controller ``0.2.0`` package，
复制 RLinf source，运行 ``requirements/install.sh f1-realworld``，并在 build 中
import ROS message packages、Ray、RLinf、Controller、Pillow、ImageIO 和 Torch。
对于 Docker，``F1_ROBOT_CONTROLLER_PACKAGE_URL`` 必须是 build 能获取的
package URL 或 pinned git URL；Dockerfile 不会把 host-local wheel copy 进
image。对于本地 ``requirements/install.sh f1-realworld``，
``F1_ROBOT_CONTROLLER_PACKAGE`` 可以指向 local wheel path、package URL 或
pinned git URL。它们是分发输入，不是 runtime approval token 或 safety
artifact。

使用默认 entrypoint。它会 source ``/opt/ros/jazzy/setup.bash``，激活
``/opt/rlinf-venv``，然后执行你的命令。

Edit the Runtime YAML
---------------------

使用一个 YAML 作为 runtime contract。真机配置是：
``examples/embodiment/config/realworld_f1_peg_rlpd_cnn_async.yaml``。

保持这些字段内联：

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

What this does: ``controller.backend`` 选择 Controller implementation；
``action_scale`` 把 normalized policy actions 转成 physical deltas；
``motion_envelope`` 限制 Controller command。用普通 Hydra key 覆盖 model 和
log paths，例如 ``actor.model.model_path=/models/f1-cnn`` 和
``runner.logger.log_path=/runs/f1-peg``。

Run Controller Diagnostics
--------------------------

真机运行前在 Thor 上运行：

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

What this does: 你仍然只编辑一个 Hydra YAML，但 Controller CLI 接收它自己拥有
的临时 controller JSON：提取出的 ``controller`` mapping 加 sibling
``motion_envelope``。``f1-controller doctor`` 检查 ROS 2 discovery、required
topics、message types、observation freshness、observation skew 和 read-only
connectivity。``f1-controller read-state`` 打印 RLinf reset 将作为 session
origin 使用的 normalized current robot state。

Start the Ray Cluster
---------------------

先启动 GPU head：

.. code-block:: bash

   export F1_RUNTIME_ROLE=gpu
   export F1_VENV_PATH="$PWD/.venv-f1"
   export RLINF_COMM_NET_DEVICES=<GPU_INTERFACE>
   export RLINF_NODE_RANK=0
   source ray_utils/realworld/f1/setup_before_ray.sh
   ray start --head --port=6379 --node-ip-address=<GPU_SERVER_IP>

在 container 中启动 Thor worker：

.. code-block:: bash

   docker run --rm --network host --name f1-ray-worker f1-rlinf-env:runtime-simplified bash -lc '
     export F1_RUNTIME_ROLE=thor
     export F1_VENV_PATH=/opt/rlinf-venv
     export RLINF_COMM_NET_DEVICES=<THOR_INTERFACE>
     export RLINF_NODE_RANK=1
     source ray_utils/realworld/f1/setup_before_ray.sh
     ray start --address=<GPU_SERVER_IP>:6379 --node-ip-address=<THOR_IP> --block
   '

What this does: setup script 验证 Python ``3.12.3``、Ray ``2.57.0``、选定角色
和通信接口。在 Thor 上还会 source ROS 2，并设置
``RMW_IMPLEMENTATION=rmw_cyclonedds_cpp``。

Run the Fake Smoke
------------------

在 GPU server 上运行软件 smoke：

.. code-block:: bash

   EMBODIED_PATH="$PWD/examples/embodiment" \
   python examples/embodiment/train_async.py \
     --config-name realworld_dummy_f1_peg_sac_cnn_async

What this does: 使用 ``controller.backend: fake``，同时保留相同 Env action
schema 和 10-step horizon。每次真机 session 前先运行它，用来捕获 Python、
Hydra、replay 和 training-chain 问题。

Run One-Step Real Motion Smoke
------------------------------

只在 operator confirmation 后运行一个真实 command。确认工作区清空，急停可触达，
机器人处于预期起始姿态，并且有人准备好停止 motion。

.. code-block:: bash

   EMBODIED_PATH="$PWD/examples/embodiment" \
   python examples/embodiment/train_async.py \
     --config-name realworld_f1_peg_rlpd_cnn_async \
     runner.max_steps=1 \
     env.train.max_episode_steps=1 \
     env.train.override_cfg.max_num_steps=1 \
     algorithm.replay_buffer.auto_save=false

What this does: entry script 只在 GPU head 运行。Ray 把 Env Worker 调度到
Thor，reset 读取当前 robot state，一个 Controller command 受
``motion_envelope`` 约束。

Run the 10-Step Async Chain
---------------------------

从 GPU head 启动受限训练链路：

.. code-block:: bash

   EMBODIED_PATH="$PWD/examples/embodiment" \
   python examples/embodiment/train_async.py \
     --config-name realworld_f1_peg_rlpd_cnn_async

What this does: RLinf 运行一个 10-step episode，把有效 transitions 写入 online
replay buffer，按配置采样 demo data，执行 SAC/RLPD update，并把 actor weights
同步回 rollout。

Audit Logs, Replay, and Stop Safely
-----------------------------------

重复运行前先检查本次结果。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 项目
     - 命令或检查
   * - Ray cluster
     - 真机运行前，在 GPU head 上用 ``ray status`` 确认有两个 live nodes。
   * - Training logs
     - 打开 ``runner.logger.log_path`` 下的 TensorBoard，检查 replay counts、SAC losses 和 timing metrics。
   * - Replay audit
     - ``python toolkits/f1/validate_f1_dataset.py --dataset ./logs/f1-peg-rlpd/online-replay``
   * - Safety stop
     - 用 ``Ctrl+C`` 停止 runner。在 GPU server 上运行 ``ray stop``，并停止 Thor container。如果 Controller 报 fault，先停止 robot motion 再收集 logs。

Troubleshooting
---------------

修改 YAML 前先检查这些项：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 现象
     - 检查
   * - EnvGroup 落到 GPU 节点
     - 确认 Thor 执行 ``ray start`` 前已经设置 ``RLINF_NODE_RANK=1``。
   * - Ray 节点互相不可见
     - 确认两个节点使用同一个可路由 ``RLINF_COMM_NET_DEVICES`` 接口，并且 Thor container 使用 host networking。
   * - Controller diagnostics 失败
     - 先修复 ROS 2 discovery、topic names、message types 或 stale sensors，再运行 RLinf。
   * - Replay validation 失败
     - 把 trajectory 视为无效。检查 audit fields，并在下一次真机尝试前重新运行 Fake smoke。
