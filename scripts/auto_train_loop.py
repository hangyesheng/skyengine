"""
自动化训练循环脚本

功能：
1. 直接调用 executor 层运行训练（不走 HTTP）
2. 随机读取 dataset/agv-instances 下的数据文件，生成配置
3. 调用 bootstrap() 创建环境和 Agent，运行训练循环
4. 直接从 Agent 对象收集训练指标（reward, loss, epsilon 等）
5. 重复直到达到结束条件（reward 阈值、loss 阈值、或最大迭代次数）

收敛检测条件（可组合，所有启用的条件都必须满足）：
- reward 阈值：当 episode reward >= 阈值时判定收敛
- loss 阈值：当所有 loss 指标 < 阈值时判定收敛
- 收敛耐心度：连续满足所有条件的迭代次数
- reward 稳定性：最近 N 次迭代的 reward 变异系数(CV)低于阈值时判定收敛
- epsilon 下界：当 epsilon <= 阈值时判定收敛
- Makespan Gap：当平均 makespan gap < 阈值时判定收敛

用法：
    uv run python scripts/auto_train_loop.py [选项]

示例：
    python scripts/auto_train_loop.py
    python scripts/auto_train_loop.py --agent GraphDualAgent
    python scripts/auto_train_loop.py --agent GraphDPAgent --log-level DEBUG
    python scripts/auto_train_loop.py --log-level WARNING --reward-threshold 500
    python scripts/auto_train_loop.py --max-iterations 100 --train-interval 20
    python scripts/auto_train_loop.py --loss-threshold 0.01 --patience 3
    python scripts/auto_train_loop.py --reward-threshold 500 --loss-threshold 0.1 --patience 2
    python scripts/auto_train_loop.py --reward-stability-window 10 --reward-stability-threshold 0.05
    python scripts/auto_train_loop.py --epsilon-floor 0.05

中断后续跑：
    再次运行相同命令即可，已完成的迭代会自动跳过（基于结果文件检测）
"""
import copy
import random
import time
import yaml
import json
import sys
import signal
import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# === 路径常量 ===
DATA_DIR = PROJECT_ROOT / "dataset" / "agv-instances"
TRAINING_LOGS = PROJECT_ROOT / "training_logs"
RESULTS_DIR = TRAINING_LOGS / "results"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "application"
    / "backend"
    / "packet_factory"
    / "config"
    / "application_config.yaml"
)
AGENTS_CONFIG_DIR = (
    PROJECT_ROOT
    / "application"
    / "backend"
    / "packet_factory"
    / "config"
    / "agents"
)

# 可选的 Agent 类型
AVAILABLE_AGENTS = [
    "GraphGRPOAgent", "GraphPPOAgent", "GraphDualAgent", "GraphDPAgent",
    "DualDRLAgent", "ORToolsAgent", "ORToolsBatchAgent",
]

# DRL agents（需要训练更新）
DRL_AGENTS = {
    "DualDRLAgent", "GraphDPAgent", "GraphDualAgent",
    "GraphPPOAgent", "GraphGRPOAgent",
}

# 结束条件配置
REWARD_THRESHOLD = 1000.0  # 累积 reward 阈值
LOSS_THRESHOLD = None  # Loss 阈值（None 表示不检查）
CONVERGENCE_PATIENCE = 1  # 收敛满足条件的连续迭代次数
MIN_ITERS = 30  # 最小迭代次数（防止假收敛）
MAKESPAN_GAP_THRESHOLD = 0.30  # Makespan Gap 早停阈值（相对下界 30%）
REWARD_STABILITY_WINDOW = None  # Reward 稳定性检测窗口（None 表示不检查）
REWARD_STABILITY_THRESHOLD = 0.05  # Reward 变异系数阈值（std/mean < 此值视为稳定）
EPSILON_FLOOR = None  # Epsilon 下界阈值（None 表示不检查）
MAX_ITERATIONS = 1000  # 最大迭代次数（防止无限循环）
TRAIN_INTERVAL = 20  # 训练间隔：每隔多少步调用一次 agent.train()
EPISODE_TIMEOUT = 7200  # 单次 episode 超时（秒）= 2小时

# === 全局中断标记 ===
_shutdown_requested = False

# 日志记录器
logger = logging.getLogger("auto_train_loop")


def setup_logging(log_level: str = "INFO", backend_log_level: str = "WARNING"):
    """
    配置日志系统

    Args:
        log_level: 前端脚本日志级别 (DEBUG, INFO, WARNING, ERROR)
        backend_log_level: 后端 logger 日志级别 (DEBUG, INFO, WARNING, ERROR)，默认 WARNING
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }

    level = level_map.get(log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

    # 设置后端日志级别环境变量，供 executor 层的 Logger 读取
    os.environ['BACKEND_LOG_LEVEL'] = backend_log_level.upper()
    logger.debug(f"后端日志级别已设置为: {backend_log_level.upper()}")


def parse_args():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="自动化训练循环脚本（直接调用 executor 层，不走 HTTP）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/auto_train_loop.py
  python scripts/auto_train_loop.py --agent GraphDualAgent
  python scripts/auto_train_loop.py --agent GraphDPAgent --log-level DEBUG
  python scripts/auto_train_loop.py --log-level WARNING --reward-threshold 500
  python scripts/auto_train_loop.py --max-iterations 100 --train-interval 20
  python scripts/auto_train_loop.py --loss-threshold 0.01 --patience 3
  python scripts/auto_train_loop.py --reward-threshold 500 --loss-threshold 0.1 --patience 2
  python scripts/auto_train_loop.py --reward-stability-window 10 --reward-stability-threshold 0.05
  python scripts/auto_train_loop.py --epsilon-floor 0.05
  python scripts/auto_train_loop.py --reward-stability-window 10 --epsilon-floor 0.01 --patience 3
  python scripts/auto_train_loop.py --log-level WARNING --backend-log-level ERROR
        """
    )

    parser.add_argument(
        '--agent',
        type=str,
        default='GraphPPOAgent',
        choices=AVAILABLE_AGENTS,
        help=f'训练使用的 Agent 类型 (默认: GraphPPOAgent)，可选: {", ".join(AVAILABLE_AGENTS)}'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='前端脚本日志级别 (默认: INFO)'
    )

    parser.add_argument(
        '--backend-log-level',
        type=str,
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='后端服务日志级别 (默认: WARNING)'
    )

    parser.add_argument(
        '--reward-threshold',
        type=float,
        default=REWARD_THRESHOLD,
        help=f'累积 reward 阈值 (默认: {REWARD_THRESHOLD})'
    )

    parser.add_argument(
        '--loss-threshold',
        type=float,
        default=None,
        help='Loss 阈值：当所有 loss 指标低于此值时判定收敛 (默认: 不检查)'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=CONVERGENCE_PATIENCE,
        help=f'收敛耐心度：连续满足所有收敛条件的迭代次数 (默认: {CONVERGENCE_PATIENCE})'
    )

    parser.add_argument(
        '--min-iters',
        type=int,
        default=MIN_ITERS,
        help=f'最小迭代次数，低于此数不允许收敛停止 (默认: {MIN_ITERS})'
    )

    parser.add_argument(
        '--makespan-gap-threshold',
        type=float,
        default=MAKESPAN_GAP_THRESHOLD,
        help=f'Makespan Gap 早停阈值：相对下界偏差比例 (默认: {MAKESPAN_GAP_THRESHOLD})'
    )

    parser.add_argument(
        '--reward-stability-window',
        type=int,
        default=None,
        help=f'Reward 稳定性检测窗口：最近 N 次迭代的 reward 变异系数(CV)低于阈值时判定收敛 (默认: 不检查)'
    )

    parser.add_argument(
        '--reward-stability-threshold',
        type=float,
        default=REWARD_STABILITY_THRESHOLD,
        help=f'Reward 稳定性变异系数阈值：std/mean < 此值时视为稳定 (默认: {REWARD_STABILITY_THRESHOLD})'
    )

    parser.add_argument(
        '--epsilon-floor',
        type=float,
        default=None,
        help='Epsilon 下界阈值：当 epsilon <= 此值时判定收敛 (默认: 不检查)'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=MAX_ITERATIONS,
        help=f'最大迭代次数 (默认: {MAX_ITERATIONS})'
    )

    parser.add_argument(
        '--train-interval',
        type=int,
        default=TRAIN_INTERVAL,
        help=f'训练间隔：每隔多少步调用一次 agent.train() (默认: {TRAIN_INTERVAL})'
    )

    parser.add_argument(
        '--episode-timeout',
        type=int,
        default=EPISODE_TIMEOUT,
        help=f'单次 episode 超时（秒）(默认: {EPISODE_TIMEOUT})'
    )

    parser.add_argument(
        '--config-yaml',
        type=str,
        default=None,
        help='可选的自定义基础 YAML 配置路径'
    )

    parser.add_argument(
        '--experiment-id',
        type=str,
        default=None,
        help='实验标识（默认自动生成时间戳），结果保存在 training_logs/results/{experiment_id}/'
    )

    return parser.parse_args()


def load_agent_config(agent_name: str) -> dict:
    """
    从 config/agents/ 目录加载指定 Agent 的配置文件

    Args:
        agent_name: Agent 名称（如 GraphPPOAgent, GraphDualAgent）

    Returns:
        dict: Agent 配置字典
    """
    config_path = AGENTS_CONFIG_DIR / f"{agent_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Agent 配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)
    logger.info(f"已加载 Agent 配置: {config_path}")
    return agent_config


# ==================== AGV 实例解析 ====================

def parse_agv_instance(filepath: Path) -> dict:
    """
    解析 AGV 数据文件

    格式说明：
    - 第一行: jobs machines agvs points links
    - 接下来 N 行: 作业工序数据
    - 接下来 P 行: 节点信息 (point_id, x, y)
    - 接下来 L 行: 边信息 (link_id, begin, end, weight)
    - 接下来 M 行: 机器绑定 (machine_id, point_id)
    - 最后 A 行: AGV配置 (agv_id, point_id, velocity)

    Returns:
        dict: 包含 jobs, points, links, machines, agvs 的字典
    """
    with open(filepath, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    idx = 0

    # 解析第一行
    first_line = lines[idx].strip().split()
    job_count = int(first_line[0])
    machine_count = int(first_line[1])
    agv_count = int(first_line[2])
    point_count = int(first_line[3])
    link_count = int(first_line[4])
    idx += 1

    # 解析作业数据
    jobs = []
    for job_idx in range(job_count):
        if idx >= len(lines):
            raise ValueError(f"文件过早结束，期望 {job_count} 个作业，但只找到 {job_idx} 个")

        ops_raw = list(map(int, lines[idx].strip().split()))
        idx += 1

        if len(ops_raw) < 1:
            raise ValueError(f"作业 {job_idx} 的数据行为空")

        ops = []
        ptr = 1
        num_operations = ops_raw[0]

        for _ in range(num_operations):
            if ptr >= len(ops_raw):
                raise ValueError(f"作业 {job_idx} 的工序 {len(ops)} 数据不完整")

            choose_machine_count = ops_raw[ptr]
            ptr += 1

            machines = []
            for _ in range(choose_machine_count):
                if ptr + 1 >= len(ops_raw):
                    raise ValueError(f"作业 {job_idx} 的工序 {len(ops)} 的机器选项数据不完整")

                machine_id = ops_raw[ptr]
                duration = ops_raw[ptr + 1]
                machines.append((machine_id, duration))
                ptr += 2

            ops.append(machines)

        jobs.append((job_idx, ops))

    # 解析 Points
    points = []
    for _ in range(point_count):
        if idx >= len(lines):
            raise ValueError(f"文件过早结束，期望 {point_count} 个节点，但只找到 {len(points)} 个")
        parts = lines[idx].strip().split()
        if len(parts) != 3:
            raise ValueError(f"节点数据格式错误（第 {idx+1} 行）: {lines[idx].strip()}")
        point_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        points.append((point_id, x, y))
        idx += 1

    # 解析 Links
    links = []
    for _ in range(link_count):
        if idx >= len(lines):
            raise ValueError(f"文件过早结束，期望 {link_count} 条边，但只找到 {len(links)} 条")
        parts = lines[idx].strip().split()
        if len(parts) != 4:
            raise ValueError(f"边数据格式错误（第 {idx+1} 行）: {lines[idx].strip()}")
        link_id = int(parts[0])
        point1_id = int(parts[1])
        point2_id = int(parts[2])
        weight = float(parts[3])
        links.append((link_id, point1_id, point2_id, weight))
        idx += 1

    # 解析 Machines
    machines = []
    for _ in range(machine_count):
        if idx >= len(lines):
            raise ValueError(f"文件过早结束，期望 {machine_count} 台机器，但只找到 {len(machines)} 台")
        parts = lines[idx].strip().split()
        if len(parts) != 2:
            raise ValueError(f"机器绑定数据格式错误（第 {idx+1} 行）: {lines[idx].strip()}")
        machine_id = int(parts[0])
        point_id = int(parts[1])
        machines.append((machine_id, point_id))
        idx += 1

    # 解析 AGVs
    agvs = []
    for _ in range(agv_count):
        if idx >= len(lines):
            raise ValueError(f"文件过早结束，期望 {agv_count} 个AGV，但只找到 {len(agvs)} 个")
        parts = lines[idx].strip().split()
        if len(parts) != 3:
            raise ValueError(f"AGV配置数据格式错误（第 {idx+1} 行）: {lines[idx].strip()}")
        agv_id = int(parts[0])
        point_id = int(parts[1])
        velocity = float(parts[2])
        agvs.append((agv_id, point_id, velocity))
        idx += 1

    return {
        "job_count": job_count,
        "machine_count": machine_count,
        "agv_count": agv_count,
        "jobs": jobs,
        "points": points,
        "links": links,
        "machines": machines,
        "agvs": agvs
    }


# ==================== 配置构建 ====================

def generate_instance_config(parsed_data: dict) -> dict:
    """从解析的数据生成 job_config / map_config / event_config"""
    points = [
        {"point": {"id": pid, "coordinate": [x, y]}}
        for pid, x, y in parsed_data['points']
    ]
    links = [
        {"link": {"id": lid, "begin": p1, "end": p2}}
        for lid, p1, p2, _ in parsed_data['links']
    ]
    machines = [
        {"machine": {"id": mid, "type": "packet_factory.Machine", "point_id": pid}}
        for mid, pid in parsed_data['machines']
    ]
    agvs = [
        {"agv": {"id": aid, "type": "packet_factory.Agv", "point_id": pid, "velocity": vel, "capacity": 12}}
        for aid, pid, vel in parsed_data['agvs']
    ]

    all_x = [p[1] for p in parsed_data['points']]
    all_y = [p[2] for p in parsed_data['points']]
    width = int(max(all_x) + 5) if all_x else 20
    height = int(max(all_y) + 5) if all_y else 30

    jobs_yaml = []
    for job_id, operations in parsed_data["jobs"]:
        job_entry = {"job": {"id": job_id, "operations": []}}
        for op_idx, machine_options in enumerate(operations):
            op_entry = {
                "operation": {
                    "id": op_idx,
                    "machines": [{"id": m, "time": d} for m, d in machine_options]
                }
            }
            job_entry["job"]["operations"].append(op_entry)
        jobs_yaml.append(job_entry)

    return {
        "event_config": {
            "event_type": [
                "packet_factory.JUST_TEST",
                "packet_factory.ENV_PAUSED",
                "packet_factory.ENV_RECOVER",
                "packet_factory.ENV_RESTART",
                "packet_factory.AGV_FAIL",
                "packet_factory.MACHINE_FAIL",
                "packet_factory.JOB_ADD"
            ]
        },
        "job_config": {"jobs": jobs_yaml},
        "map_config": {
            "width": width,
            "height": height,
            "points": points,
            "machines": machines,
            "links": links,
            "agvs": agvs
        },
    }


def build_config(agent_key: str, instance_config: dict,
                 base_yaml_path: Optional[str] = None) -> dict:
    """构建完整的 bootstrap 配置

    从 config/agents/{agent_key}.yaml 加载 Agent 超参数，
    设置 ui_mode=backend, task_mode=training。

    Args:
        agent_key: Agent 标识键
        instance_config: generate_instance_config() 的输出
        base_yaml_path: 可选的自定义基础 YAML 路径
    """
    yaml_path = base_yaml_path or str(DEFAULT_CONFIG_PATH)
    with open(yaml_path, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    config = copy.deepcopy(template["config"])

    # 从 per-agent 配置文件加载完整 agent 参数（超参数 + identity 字段，含 mode）
    agent_full_config = load_agent_config(agent_key)

    # 填充 agent 段
    config["simulation"]["agent"].update(agent_full_config)

    # 训练模式：始终使用 backend + training
    config["simulation"]["ui_mode"] = "backend"
    config["simulation"]["task_mode"] = "training"

    # 注入实例数据
    config["simulation"]["job_config"] = instance_config["job_config"]
    config["simulation"]["map_config"] = instance_config["map_config"]
    config["simulation"]["event_config"] = instance_config["event_config"]

    return config


# ==================== Episode 运行 ====================

def run_training_episode(config: dict, train_interval: int = TRAIN_INTERVAL,
                         timeout: int = EPISODE_TIMEOUT) -> dict:
    """运行一次完整训练 episode 并返回结果

    直接调用 executor 层，不走 HTTP。
    训练逻辑与 BackendCore._run_backend_training() 一致。

    Args:
        config: bootstrap 配置
        train_interval: 训练间隔步数
        timeout: 单次 episode 超时秒数

    Returns:
        dict: 包含 status, makespan, metrics, steps, elapsed 等字段
    """
    from executor.packet_factory.lifecycle.bootstrap import bootstrap
    from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import EnvStatus

    env, agent = bootstrap(config)

    # 重置环境（加载实例数据，初始化 jobs/machines/agvs）
    env.reset()

    # headless 模式：设置环境状态为 RUNNING，禁用可视化
    env.status = EnvStatus.RUNNING
    if env.env_visualizer is not None:
        env.env_visualizer = None

    start_time = time.time()
    step_count = 0

    try:
        while not env.env_is_finished():
            if _shutdown_requested:
                logger.info("收到中断信号，正在停止当前 episode...")
                metrics = agent.get_training_metrics() if hasattr(agent, 'get_training_metrics') else {}
                return {
                    "status": "interrupted",
                    "makespan": env.env_timeline,
                    "metrics": metrics,
                    "decision_stats": agent.get_decision_stats() if hasattr(agent, 'get_decision_stats') else {},
                    "steps": step_count,
                    "elapsed": time.time() - start_time,
                }

            if time.time() - start_time > timeout:
                logger.warning(f"Episode 超时 ({timeout}s)，已执行 {step_count} 步")
                metrics = agent.get_training_metrics() if hasattr(agent, 'get_training_metrics') else {}
                return {
                    "status": "timeout",
                    "makespan": env.env_timeline,
                    "metrics": metrics,
                    "decision_stats": agent.get_decision_stats() if hasattr(agent, 'get_decision_stats') else {},
                    "steps": step_count,
                    "elapsed": time.time() - start_time,
                }

            step_count += 1

            # 输入获得环境状态并决策
            actions = env.action_space(agent)

            # 执行动作
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # 训练更新（每 train_interval 步）
            if step_count % train_interval == 0:
                if hasattr(agent, 'update'):
                    agent.update(observations, rewards)
                elif hasattr(agent, 'train'):
                    agent.train(observations, rewards, terminations, truncations, infos)

            # 主动释放 GIL
            time.sleep(0)

            # 每 50 步打印训练指标
            if step_count % 50 == 0 and hasattr(agent, 'get_training_metrics'):
                metrics = agent.get_training_metrics()
                ep_reward = metrics.get('episode_reward', 0.0)
                epsilon = metrics.get('epsilon', 'N/A')
                loss_info = {k: f"{v:.6f}" for k, v in metrics.items()
                             if 'loss' in k.lower() and isinstance(v, (int, float))}
                logger.info(f"[Step {step_count}] reward={ep_reward:.4f}, "
                            f"epsilon={epsilon}, timeline={env.env_timeline}"
                            + (f", {loss_info}" if loss_info else ""))

        makespan = env.env_timeline
        metrics = agent.get_training_metrics() if hasattr(agent, 'get_training_metrics') else {}
        decision_stats = agent.get_decision_stats() if hasattr(agent, 'get_decision_stats') else {}
        elapsed = time.time() - start_time

        return {
            "status": "completed",
            "makespan": makespan,
            "metrics": metrics,
            "decision_stats": decision_stats,
            "steps": step_count,
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.error(f"Episode 运行异常: {e}")
        import traceback
        traceback.print_exc()
        metrics = agent.get_training_metrics() if hasattr(agent, 'get_training_metrics') else {}
        return {
            "status": "error",
            "makespan": env.env_timeline if hasattr(env, 'env_timeline') else 0.0,
            "metrics": metrics,
            "decision_stats": {},
            "steps": step_count,
            "elapsed": time.time() - start_time,
            "error": str(e),
        }


def save_training_result(agent, env, result: dict, experiment_dir: Path):
    """保存训练结果

    复刻 BackendCore._save_training_results() 的逻辑，
    保存到实验目录下的 training_report.json。

    Args:
        agent: Agent 实例
        env: 环境实例
        result: run_training_episode() 的返回值
        experiment_dir: 实验结果目录
    """
    try:
        # 优先调用 Agent 的 save_training_result 方法（如果存在）
        if hasattr(agent, 'save_training_result'):
            result_path = agent.save_training_result()
            if result_path:
                logger.info(f"Agent 保存训练结果到: {result_path}")
        else:
            # 降级方案：保存到实验目录
            agent_name = getattr(agent, 'name', 'UnknownAgent')
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            result_subdir = experiment_dir / f"{agent_name}_{timestamp}"
            result_subdir.mkdir(parents=True, exist_ok=True)

            results = {
                'makespan': result.get('makespan', env.env_timeline if hasattr(env, 'env_timeline') else 0),
                'decision_stats': result.get('decision_stats', {}),
                'q_table_size': len(getattr(agent, 'q_table', {})),
                'training_metrics': result.get('metrics', {}),
                'metadata': {
                    'agent_name': agent_name,
                    'agent_id': getattr(agent, 'agent_id', None),
                    'save_time': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            result_file = result_subdir / 'training_report.json'
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"训练结果已保存到: {result_file}")

        # 保存模型（使用 Agent 的默认路径或降级方案）
        if hasattr(agent, 'save_model'):
            model_path = agent.save_model()
            if model_path:
                logger.info(f"模型已保存到: {model_path}")

    except Exception as e:
        logger.error(f"保存训练结果失败: {e}")


# ==================== 收敛检测 ====================

def compute_makespan_lower_bound(parsed_data: dict) -> float:
    """计算估计的 makespan 下界

    使用两个下界取最大值：
    1. 作业下界：最长作业的最小总处理时间
    2. 机器下界：最小总工作量 / 机器数

    Returns:
        float: 估计的下界 (> 0)
    """
    job_min_times = []
    total_min_work = 0.0
    for job_id, operations in parsed_data['jobs']:
        if not operations:
            continue
        job_time = sum(min(d for _, d in op) for op in operations if op)
        job_min_times.append(job_time)
        for op in operations:
            if op:
                total_min_work += min(d for _, d in op)

    job_lb = max(job_min_times) if job_min_times else 0.0
    machine_lb = total_min_work / max(parsed_data['machine_count'], 1)

    lb = max(job_lb, machine_lb)
    return max(lb, 1.0)


def check_convergence(iteration_metrics: List[Dict], reward_threshold: float,
                      loss_threshold: Optional[float], patience: int,
                      min_iters: int = 30, current_iteration: int = 0,
                      makespan_gap_threshold: float = 0.30,
                      reward_stability_window: Optional[int] = None,
                      reward_stability_threshold: float = 0.05,
                      epsilon_floor: Optional[float] = None) -> Tuple[bool, str]:
    """检查训练是否收敛（含 min_iters 保护 + Makespan Gap 早停 + Reward 稳定性 + Epsilon 下界）

    收敛条件（所有启用的条件都必须满足）：
    1. current_iteration >= min_iters（防止假收敛）
    2. episode_reward >= reward_threshold 连续 patience 次（如果 threshold > 0）
    3. 所有 loss 指标 < loss_threshold 连续 patience 次（如果启用）
    4. 平均 Makespan Gap < makespan_gap_threshold 连续 patience 次（如果启用）
    5. 最近 reward_stability_window 次迭代 reward 变异系数 < reward_stability_threshold（如果启用）
    6. epsilon <= epsilon_floor（如果启用）

    Args:
        iteration_metrics: 历次迭代的 metrics 列表
        reward_threshold: reward 阈值（0 表示不检查）
        loss_threshold: loss 阈值（None 表示不检查）
        patience: 连续满足条件的迭代次数
        min_iters: 最小迭代次数
        current_iteration: 当前迭代编号
        makespan_gap_threshold: Makespan Gap 阈值（0 表示不检查）
        reward_stability_window: Reward 稳定性检测窗口大小（None 表示不检查）
        reward_stability_threshold: Reward 变异系数阈值（std/mean）
        epsilon_floor: Epsilon 下界阈值（None 表示不检查）

    Returns:
        Tuple[bool, str]: (是否收敛, 收敛原因描述)
    """
    if not iteration_metrics:
        return False, ""

    # min_iters 保护
    if current_iteration < min_iters:
        return False, f"未达到最小迭代次数 ({current_iteration}/{min_iters})"

    # 检查最近 patience 次迭代
    recent = iteration_metrics[-patience:] if len(iteration_metrics) >= patience else iteration_metrics
    if len(recent) < patience:
        return False, ""

    # 所有 recent 迭代都需要满足条件
    all_reward_ok = True
    all_loss_ok = True
    all_makespan_gap_ok = True
    loss_details = []

    # 计算 Makespan Gap 统计
    gaps = []
    for m in recent:
        reward = m.get('episode_reward', 0.0)
        if reward_threshold > 0 and reward < reward_threshold:
            all_reward_ok = False

        if loss_threshold is not None:
            for key, value in m.items():
                if 'loss' in key.lower() and isinstance(value, (int, float)):
                    if value >= loss_threshold:
                        all_loss_ok = False
                        loss_details.append(f"{key}={value:.6f}")

        # Makespan Gap 检查
        if makespan_gap_threshold > 0:
            makespan = m.get('makespan', 0.0)
            lower_bound = m.get('makespan_lower_bound', 0.0)
            if lower_bound > 0 and makespan > 0:
                gap = (makespan - lower_bound) / lower_bound
                gaps.append(gap)

    # Makespan Gap 使用平均值判断（适应不同实例规模）
    avg_gap = 0.0
    if makespan_gap_threshold > 0 and gaps:
        avg_gap = sum(gaps) / len(gaps)
        if avg_gap > makespan_gap_threshold:
            all_makespan_gap_ok = False

    # Reward 稳定性检测（最近 window 次迭代的变异系数）
    reward_stable = True
    reward_cv = None
    if reward_stability_window is not None and len(iteration_metrics) >= reward_stability_window:
        recent_rewards = [m.get('episode_reward', 0.0) for m in iteration_metrics[-reward_stability_window:]]
        mean_r = sum(recent_rewards) / len(recent_rewards)
        if abs(mean_r) > 1e-8:
            std_r = (sum((r - mean_r) ** 2 for r in recent_rewards) / len(recent_rewards)) ** 0.5
            reward_cv = std_r / abs(mean_r)
            if reward_cv >= reward_stability_threshold:
                reward_stable = False
        elif all(abs(r) < 1e-8 for r in recent_rewards):
            # 所有 reward 都为 0，不算稳定
            reward_stable = False

    # Epsilon 下界检测
    epsilon_at_floor = True
    latest_epsilon = None
    if epsilon_floor is not None:
        latest_epsilon = iteration_metrics[-1].get('epsilon', None)
        if latest_epsilon is None or latest_epsilon > epsilon_floor:
            epsilon_at_floor = False

    reasons = []
    if all_reward_ok and reward_threshold > 0:
        reasons.append(f"reward >= {reward_threshold}")
    if all_loss_ok and loss_threshold is not None:
        reasons.append(f"all losses < {loss_threshold}")
    if all_makespan_gap_ok and makespan_gap_threshold > 0 and gaps:
        reasons.append(f"avg makespan gap {avg_gap:.1%} < {makespan_gap_threshold:.1%}")
    if reward_stable and reward_stability_window is not None and reward_cv is not None:
        reasons.append(f"reward stable (CV={reward_cv:.4f} < {reward_stability_threshold})")
    if epsilon_at_floor and epsilon_floor is not None and latest_epsilon is not None:
        reasons.append(f"epsilon={latest_epsilon:.6f} <= {epsilon_floor}")

    # 所有启用的条件都满足才收敛
    conditions = []
    if reward_threshold > 0:
        conditions.append(all_reward_ok)
    if loss_threshold is not None:
        conditions.append(all_loss_ok)
    if makespan_gap_threshold > 0:
        conditions.append(all_makespan_gap_ok)
    if reward_stability_window is not None:
        conditions.append(reward_stable)
    if epsilon_floor is not None:
        conditions.append(epsilon_at_floor)

    converged = all(conditions) if conditions else False

    if converged:
        reason = ", ".join(reasons)
        return True, f"连续 {patience} 次满足: {reason}"

    # 提供不收敛的具体原因
    if not all_makespan_gap_ok:
        return False, f"makespan gap {avg_gap:.1%} > {makespan_gap_threshold:.1%}"
    if not reward_stable and reward_cv is not None:
        return False, f"reward not stable (CV={reward_cv:.4f} >= {reward_stability_threshold})"
    if not epsilon_at_floor and latest_epsilon is not None:
        return False, f"epsilon={latest_epsilon:.6f} > {epsilon_floor}"

    return False, ""


# ==================== 数据集发现 ====================

def select_random_data_file() -> Optional[Path]:
    """
    随机选择一个数据文件（递归搜索所有子文件夹）

    Returns:
        Optional[Path]: 数据文件路径，如果没有找到则返回 None
    """
    data_files = list(DATA_DIR.glob('**/*_agv.txt'))
    if data_files:
        selected_file = random.choice(data_files)
        # 显示相对于 DATA_DIR 的路径，方便用户识别
        relative_path = selected_file.relative_to(DATA_DIR)
        logger.info(f"从 {len(data_files)} 个文件中随机选择: {relative_path}")
        return selected_file
    return None


# ==================== 结果管理 ====================

def load_completed_iterations(experiment_dir: Path) -> int:
    """读取已有结果，返回已完成的迭代数

    Args:
        experiment_dir: 实验目录

    Returns:
        int: 已完成的迭代数
    """
    progress_file = experiment_dir / "progress.jsonl"
    if not progress_file.exists():
        return 0

    count = 0
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
    return count


def append_iteration_result(experiment_dir: Path, record: dict):
    """追加一条迭代结果到 JSONL 文件

    Args:
        experiment_dir: 实验目录
        record: 迭代结果字典
    """
    progress_file = experiment_dir / "progress.jsonl"
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==================== 主训练循环 ====================

def main():
    """主函数"""
    args = parse_args()

    # 设置日志级别
    setup_logging(args.log_level, args.backend_log_level)

    # 确定实验 ID 和目录
    experiment_id = args.experiment_id or time.strftime("train_%Y%m%d_%H%M%S")
    experiment_dir = RESULTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 加载 Agent 配置（仅用于验证配置文件存在）
    agent_config = load_agent_config(args.agent)

    logger.info("=" * 60)
    logger.info("自动化训练循环脚本（直接调用 executor 层）")
    logger.info(f"训练 Agent: {args.agent}")
    logger.info(f"Reward 阈值: {args.reward_threshold:.2f}")
    logger.info(f"Loss 阈值: {args.loss_threshold if args.loss_threshold is not None else '不检查'}")
    logger.info(f"收敛耐心度: {args.patience}")
    logger.info(f"最小迭代次数: {args.min_iters}")
    logger.info(f"Makespan Gap 阈值: {args.makespan_gap_threshold * 100:.1f}%")
    logger.info(f"Reward 稳定性: window={args.reward_stability_window or '不检查'}, "
                f"CV阈值={args.reward_stability_threshold}")
    logger.info(f"Epsilon 下界: {args.epsilon_floor if args.epsilon_floor is not None else '不检查'}")
    logger.info(f"最大迭代次数: {args.max_iterations}")
    logger.info(f"训练间隔: 每 {args.train_interval} 步")
    logger.info(f"Episode 超时: {args.episode_timeout}s")
    logger.info(f"前端脚本日志级别: {args.log_level.upper()}")
    logger.info(f"后端服务日志级别: {args.backend_log_level.upper()}")
    logger.info(f"实验目录: {experiment_dir}")
    logger.info("=" * 60)

    # 保存实验配置快照
    config_snapshot = {
        "experiment_id": experiment_id,
        "agent": args.agent,
        "reward_threshold": args.reward_threshold,
        "loss_threshold": args.loss_threshold,
        "patience": args.patience,
        "min_iters": args.min_iters,
        "makespan_gap_threshold": args.makespan_gap_threshold,
        "reward_stability_window": args.reward_stability_window,
        "reward_stability_threshold": args.reward_stability_threshold,
        "epsilon_floor": args.epsilon_floor,
        "max_iterations": args.max_iterations,
        "train_interval": args.train_interval,
        "episode_timeout": args.episode_timeout,
        "config_yaml": args.config_yaml,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(experiment_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    # 注册信号处理器
    def signal_handler(sig, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info("\n收到中断信号，等待当前 episode 完成后退出...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 跨迭代指标追踪
    iteration_metrics: List[Dict] = []

    # 加载已完成的迭代数（支持续跑）
    completed_iters = load_completed_iterations(experiment_dir)
    if completed_iters > 0:
        logger.info(f"检测到已完成的 {completed_iters} 次迭代，将从第 {completed_iters} 次继续")

    try:
        iteration = 0
        while iteration < args.max_iterations:
            # 检查是否达到收敛条件
            if iteration > 0 and iteration_metrics:
                converged, reason = check_convergence(
                    iteration_metrics, args.reward_threshold, args.loss_threshold, args.patience,
                    min_iters=args.min_iters, current_iteration=iteration,
                    makespan_gap_threshold=args.makespan_gap_threshold,
                    reward_stability_window=args.reward_stability_window,
                    reward_stability_threshold=args.reward_stability_threshold,
                    epsilon_floor=args.epsilon_floor,
                )
                if converged:
                    logger.info(f"\n=== 训练收敛，停止迭代 ===")
                    logger.info(f"收敛原因: {reason}")
                    break
                else:
                    if reason:
                        logger.info(f"未收敛: {reason}")

            if _shutdown_requested:
                break

            # 随机选择数据文件
            data_file = select_random_data_file()
            if not data_file:
                logger.error("未找到数据文件")
                break

            # 解析实例
            relative_path = data_file.relative_to(DATA_DIR)
            logger.info("\n" + "=" * 60)
            logger.info(f"[迭代 {iteration}] 使用数据文件: {relative_path}")
            logger.info(f"{'=' * 60}")

            try:
                parsed_data = parse_agv_instance(data_file)
                logger.info(f"解析完成: {parsed_data['job_count']} jobs, "
                            f"{parsed_data['machine_count']} machines, "
                            f"{parsed_data['agv_count']} AGVs")
            except Exception as e:
                logger.error(f"解析数据文件失败: {e}")
                iteration += 1
                continue

            # 计算当前实例的 makespan 下界（用于 Makespan Gap 早停）
            try:
                lower_bound = compute_makespan_lower_bound(parsed_data)
            except Exception:
                lower_bound = 1.0

            # 生成实例配置并构建完整配置
            instance_config = generate_instance_config(parsed_data)
            config = build_config(args.agent, instance_config,
                                  base_yaml_path=args.config_yaml)

            # 运行训练 episode
            logger.info("开始训练 episode...")
            result = run_training_episode(
                config,
                train_interval=args.train_interval,
                timeout=args.episode_timeout,
            )

            # 提取指标
            metrics = result.get("metrics", {})
            metrics['makespan'] = result.get("makespan", 0.0)
            metrics['makespan_lower_bound'] = lower_bound
            metrics['makespan_gap'] = (
                (metrics['makespan'] - lower_bound) / lower_bound
                if lower_bound > 0 else float('inf')
            )
            metrics['iteration_success'] = result.get("status") == "completed"
            metrics['steps'] = result.get("steps", 0)
            metrics['elapsed'] = result.get("elapsed", 0.0)
            iteration_metrics.append(metrics)

            # 记录迭代结果
            record = {
                "iteration": iteration,
                "data_file": str(relative_path),
                "agent": args.agent,
                "status": result.get("status", "unknown"),
                "makespan": result.get("makespan", 0.0),
                "makespan_lower_bound": lower_bound,
                "makespan_gap": metrics['makespan_gap'],
                "steps": result.get("steps", 0),
                "elapsed": result.get("elapsed", 0.0),
                "training_metrics": metrics,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if "error" in result:
                record["error"] = result["error"]

            append_iteration_result(experiment_dir, record)

            # 输出本次迭代的关键指标
            episode_reward = metrics.get('episode_reward', 0.0)
            epsilon = metrics.get('epsilon', 'N/A')
            makespan_gap = metrics.get('makespan_gap', float('inf'))
            loss_keys = {k: v for k, v in metrics.items()
                         if 'loss' in k.lower() and isinstance(v, (int, float))}

            if result.get("status") == "completed":
                logger.info(f"[迭代 {iteration}] 训练完成! "
                            f"makespan={metrics['makespan']:.2f}, "
                            f"reward={episode_reward:.4f}, "
                            f"makespan_gap={makespan_gap:.1%}, "
                            f"epsilon={epsilon}, "
                            f"steps={result.get('steps', 0)}, "
                            f"elapsed={result.get('elapsed', 0.0):.2f}s"
                            + (f", losses={{{', '.join(f'{k}={v:.6f}' for k, v in loss_keys.items())}}}"
                               if loss_keys else ""))
            else:
                logger.warning(f"[迭代 {iteration}] {result.get('status', 'unknown')}! "
                               f"makespan={metrics['makespan']:.2f}")

            iteration += 1

            # 短暂休息，避免太快
            time.sleep(1)

        if iteration >= args.max_iterations:
            logger.warning(f"\n达到最大迭代次数 {args.max_iterations}")

        # 打印最终统计
        final_metrics = iteration_metrics[-1] if iteration_metrics else {}
        final_reward = final_metrics.get('episode_reward', 0.0)
        final_losses = {k: v for k, v in final_metrics.items()
                        if 'loss' in k.lower() and isinstance(v, (int, float))}
        final_makespan = final_metrics.get('makespan', 0.0)
        final_makespan_gap = final_metrics.get('makespan_gap', float('inf'))

        logger.info("\n" + "=" * 60)
        logger.info("训练完成!")
        logger.info(f"总迭代次数: {iteration}")
        logger.info(f"最终 episode reward: {final_reward:.4f}")
        logger.info(f"最终 makespan: {final_makespan:.2f}")
        logger.info(f"最终 makespan gap: {final_makespan_gap:.1%}")
        if final_losses:
            logger.info(f"最终 losses: {{{', '.join(f'{k}={v:.6f}' for k, v in final_losses.items())}}}")
        logger.info(f"结果保存在: {experiment_dir}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\n收到键盘中断信号")
    finally:
        logger.info("训练循环已退出")


if __name__ == "__main__":
    main()
