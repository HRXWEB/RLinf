F1 双节点运行时
===============

把 F1 runtime 部署为双节点 RLinf 集群。GPU server 负责训练；机器人侧 ARM64
container 负责 ROS 2、Controller 和 Env Worker。参考部署已在 NVIDIA Thor 上
验证。两个节点都要先设置 rank 和网络接口，再加入 Ray。

运行时边界
----------

只在 GPU head 上运行 RLinf entry script。

.. list-table::
   :header-rows: 1
   :widths: 20 22 22 36

   * - 节点
     - 架构
     - Runtime
     - 职责
   * - GPU server
     - Linux x86_64
     - Host ``uv`` / venv
     - Ray head、ActorGroup、RolloutGroup、optimizer、replay sampling 和 logs。
   * - F1 机器人节点
     - Linux aarch64 / arm64
     - ROS 2 Jazzy container
     - Ray worker、EnvGroup、Controller ``backend: ros2``、sensor reads 和 bounded robot commands。

.. warning::

   在 ``ray start`` 之前设置 ``RLINF_NODE_RANK``。Ray 在启动时捕获环境变量，
   之后再修改不会修复 placement。

准备 GPU 环境
----------------

先安装 embodied dependencies，再把共享 F1 constraints 和 Controller package
安装到同一个环境：

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

What this does: 第一条命令安装 GPU training dependencies；第二条把共享 Ray
runtime 固定为 ``2.57.0``，并安装 F1 环境所需的 Controller package。GPU server
不需要 ROS 2。

构建机器人侧镜像
----------------

从同一份 RLinf source 构建 ARM64 image：

.. code-block:: bash

   export F1_ROBOT_CONTROLLER_PACKAGE_URL=<package-url-or-pinned-git-url>
   docker build --platform linux/arm64 \
     -f docker/f1/Dockerfile \
     --build-arg F1_ROBOT_CONTROLLER_PACKAGE_URL="${F1_ROBOT_CONTROLLER_PACKAGE_URL}" \
     -t f1-rlinf-env:runtime-simplified .

Dockerfile 创建 Python ``3.12.3`` venv，安装共享 F1 constraints 和用户提供的
Controller package，复制 RLinf source，并在 build 阶段验证 ROS message
packages、Ray、RLinf、Controller、Pillow、ImageIO 和 Torch imports。

``F1_ROBOT_CONTROLLER_PACKAGE_URL`` 必须是 build 能访问的 package URL 或
pinned git URL。本地运行 ``requirements/install.sh f1-realworld`` 时，
``F1_ROBOT_CONTROLLER_PACKAGE`` 也可以指向 local wheel。

默认 entrypoint 会 source ``/opt/ros/jazzy/setup.bash``、激活
``/opt/rlinf-venv``，然后执行传入的命令。

配置 Runtime YAML
-----------------

真机配置位于
``examples/embodiment/config/realworld_f1_peg_rlpd_cnn_async.yaml``。核心 placement
如下：

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
         max_num_steps: 10
         controller:
           backend: ros2
         action_scale:
           tcp_position_m: 0.005
           tcp_orientation_deg: 1.0
           gripper_percent_closed: 10.0

``action_scale`` 把 normalized policy action 转成 physical delta；
``motion_envelope`` 限制 Controller command。三个步数字段都接受正整数，请按
采集计划保持一致或明确设置各自边界。

运行 Controller 诊断
---------------------

在机器人侧节点运行：

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

What this does: 从 Hydra YAML 提取 Controller-owned mapping 和 sibling
``motion_envelope``，生成临时 controller JSON。``doctor`` 检查 ROS 2 discovery、
required topics、message types、freshness 和 skew；``read-state`` 读取 normalized
current robot state。

启动 Ray 集群
----------------

先启动 GPU head：

.. code-block:: bash

   export F1_RUNTIME_ROLE=gpu
   export F1_VENV_PATH="$PWD/.venv-f1"
   export RLINF_COMM_NET_DEVICES=<GPU_INTERFACE>
   export RLINF_NODE_RANK=0
   source ray_utils/realworld/f1/setup_before_ray.sh
   ray start --head --port=6379 --node-ip-address=<GPU_SERVER_IP>

在 container 中启动机器人侧 worker：

.. code-block:: bash

   docker run --rm --network host --name f1-ray-worker f1-rlinf-env:runtime-simplified bash -lc '
     export F1_RUNTIME_ROLE=thor
     export F1_VENV_PATH=/opt/rlinf-venv
     export RLINF_COMM_NET_DEVICES=<ROBOT_INTERFACE>
     export RLINF_NODE_RANK=1
     source ray_utils/realworld/f1/setup_before_ray.sh
     ray start --address=<GPU_SERVER_IP>:6379 --node-ip-address=<ROBOT_IP> --block
   '

setup script 验证 Python ``3.12.3``、Ray ``2.57.0``、role 和网络接口。在
机器人侧还会 source ROS 2，并设置
``RMW_IMPLEMENTATION=rmw_cyclonedds_cpp``。``thor`` 是当前 ARM64 机器人节点的
runtime role token。

准备 RLPD 数据
--------------

F1 RLPD 配置要求非空 demonstration buffer。将
``algorithm.demo_buffer.load_path`` 设置为同一任务采集的 RLinf replay buffer，
并确保 observation 和 14 维 action 的 shape 与当前配置一致。

没有 demonstration 时，删除 demo buffer 以运行 online SAC：

.. code-block:: bash

   '~algorithm.demo_buffer'

运行训练
--------

确认机器人工作区已清空，并由受过训练的操作者现场看护。然后从 GPU head 启动：

.. code-block:: bash

   EMBODIED_PATH="$PWD/examples/embodiment" \
   python examples/embodiment/train_async.py \
     --config-name realworld_f1_peg_rlpd_cnn_async \
     actor.model.model_path=/path/to/RLinf-ResNet10-pretrained \
     algorithm.demo_buffer.load_path=/path/to/f1-peg-demo-buffer

Online SAC 在命令末尾追加 ``'~algorithm.demo_buffer'``。RLinf 将 online
transition 写入 replay buffer，执行 SAC update，并将 Actor weights 同步到
Rollout。

检查与停止
----------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 项目
     - 命令或检查
   * - Ray cluster
     - 真机运行前，在 GPU head 上用 ``ray status`` 确认有两个 live nodes。
   * - Training logs
     - 打开 ``runner.logger.log_path`` 下的 TensorBoard，检查 replay counts、SAC losses 和 timing metrics。
   * - Online replay
     - 确认 ``runner.logger.log_path`` 下的 replay buffer 已写入，并且 sample count 持续增长。
   * - Safety stop
     - 用 ``Ctrl+C`` 停止 runner，在 GPU server 上运行 ``ray stop``，并停止机器人侧 container。如果 Controller 报 fault，先停止 robot motion 再收集 logs。

故障排查
--------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 现象
     - 检查
   * - EnvGroup 落到 GPU 节点
     - 确认机器人节点执行 ``ray start`` 前已经设置 ``RLINF_NODE_RANK=1``。
   * - Ray 节点互相不可见
     - 确认两个节点选择同一个可路由网络上的接口，并且机器人侧 container 使用 host networking。
   * - Controller diagnostics 失败
     - 先修复 ROS 2 discovery、topic names、message types 或 stale sensors，再运行 RLinf。
   * - Replay 没有增长
     - 确认 ``rollout.collect_transitions: true``，检查 EnvGroup 日志，并在下一次真机运行前修复 Controller 错误。
