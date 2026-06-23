"""
自动化训练循环脚本（Epoch 级别，Mini-Batch 训练）

功能：
1. 直接调用 executor 层运行训练（不走 HTTP）
2. 划分训练集和验证集，按文件夹分层采样
3. 每轮 epoch 运行 mini-batch 训练 episode，agent 权重跨 episode 持久化
4. 定期在验证集上推理评估，计算平均 makespan gap
5. 基于 epoch 级相对容差的早停机制：
   当验证指标相对改善 < tolerance 连续 patience 个 epoch 时停止
6. 支持旧版绝对收敛条件（--legacy-convergence）

架构：
- Agent 只创建一次，跨 episode 复用（神经网络权重持久化）
- 每个 episode 创建新的 env（不同实例有不同的工厂布局）
- 收敛检测基于 epoch 级验证集指标，不再是逐 episode 的绝对阈值

用法：
    uv run python scripts/auto_train_loop.py [选项]

示例：
    python scripts/auto_train_loop.py
    python scripts/auto_train_loop.py --agent GraphDualAgent
    python scripts/auto_train_loop.py --agent GraphPPOAgent --epoch-size 5 --max-epochs 50
    python scripts/auto_train_loop.py --val-ratio 0.3 --relative-tolerance 0.005
    python scripts/auto_train_loop.py --early-stop-patience 3 --min-epochs 5
    python scripts/auto_train_loop.py --resume-from training_logs/results/train_xxx
    python scripts/auto_train_loop.py --legacy-convergence --reward-threshold 500

中断后续跑：
    再次运行相同命令即可，已完成的 epoch 会自动跳过
"""
import copy
import gc
import random
import time
import yaml
import json
import sys
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

# === Epoch 级训练默认参数 ===
EPOCH_SIZE = 10           # 每个 epoch 的训练 episode 数（mini-batch 大小）
VAL_RATIO = 0.2           # 验证集比例
SPLIT_SEED = 42           # 数据集划分种子
VAL_INTERVAL = 5          # 每 N 个 epoch 验证一次（验证阶段耗时远大于训练，降低频率以提升训练效率）
EARLY_STOP_PATIENCE = 5   # 早停耐心度
RELATIVE_TOLERANCE = 0.01 # 相对容差阈值（1%）
MIN_EPOCHS = 10           # 最小 epoch 数（防止假收敛）
MAX_EPOCHS = 100          # 最大 epoch 数
TRAIN_INTERVAL = 20       # 训练间隔：每隔多少步调用一次 agent.train()
EPISODE_TIMEOUT = 7200    # 单次训练 episode 超时（秒）= 2小时

# === 验证阶段优化参数 ===
# 验证为纯推理，单实例耗时长；用更短超时 + 固定随机子集 + 死锁检测控制成本
VAL_EPISODE_TIMEOUT = 600  # 单次验证 episode 超时（秒）= 10分钟（验证只看 makespan，卡住即止损）
VAL_SAMPLE_SIZE = 10       # 每次验证随机抽取的实例数（0 = 用全部验证集）
VAL_SAMPLE_SEED = 42       # 验证子集随机种子（固定 → 跨 epoch 指标可比，早停信号稳定）
NO_PROGRESS_LIMIT = 50     # 连续 N 步 env_timeline 无推进视为死锁，立即终止该 episode

# === 旧版收敛条件默认值（仅 --legacy-convergence 时使用）===
REWARD_THRESHOLD = 1000.0
LOSS_THRESHOLD = None
CONVERGENCE_PATIENCE = 1
MAKESPAN_GAP_THRESHOLD = 0.30
REWARD_STABILITY_WINDOW = None
REWARD_STABILITY_THRESHOLD = 0.05
EPSILON_FLOOR = None

# === 配置缓存（避免每 episode 重复读取 YAML 文件）===
_cached_base_template = None       # application_config.yaml 解析后的模板
_cached_agent_config = {}          # {agent_key: dict} 缓存 Agent YAML 配置

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
        description="自动化训练循环脚本（Epoch 级别，Mini-Batch 训练）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/auto_train_loop.py
  python scripts/auto_train_loop.py --agent GraphDualAgent
  python scripts/auto_train_loop.py --agent GraphPPOAgent --epoch-size 5 --max-epochs 50
  python scripts/auto_train_loop.py --val-ratio 0.3 --relative-tolerance 0.005
  python scripts/auto_train_loop.py --early-stop-patience 3 --min-epochs 5
  python scripts/auto_train_loop.py --resume-from training_logs/results/train_xxx
  python scripts/auto_train_loop.py --legacy-convergence --reward-threshold 500
        """
    )

    # === Agent 选择 ===
    parser.add_argument(
        '--agent',
        type=str,
        default='GraphPPOAgent',
        choices=AVAILABLE_AGENTS,
        help=f'训练使用的 Agent 类型 (默认: GraphPPOAgent)，可选: {", ".join(AVAILABLE_AGENTS)}'
    )

    # === 日志级别 ===
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

    # === Epoch 级训练参数 ===
    parser.add_argument(
        '--epoch-size',
        type=int,
        default=EPOCH_SIZE,
        help=f'每个 epoch 的训练 episode 数（mini-batch 大小）(默认: {EPOCH_SIZE})'
    )

    parser.add_argument(
        '--max-epochs',
        type=int,
        default=MAX_EPOCHS,
        help=f'最大 epoch 数 (默认: {MAX_EPOCHS})'
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
        help=f'单次训练 episode 超时（秒）(默认: {EPISODE_TIMEOUT})'
    )

    parser.add_argument(
        '--val-episode-timeout',
        type=int,
        default=VAL_EPISODE_TIMEOUT,
        help=f'单次验证 episode 超时（秒），验证为纯推理故默认更短 (默认: {VAL_EPISODE_TIMEOUT})'
    )

    # === 验证集划分 ===
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=VAL_RATIO,
        help=f'验证集比例 (默认: {VAL_RATIO})'
    )

    parser.add_argument(
        '--split-seed',
        type=int,
        default=SPLIT_SEED,
        help=f'数据集划分随机种子 (默认: {SPLIT_SEED})'
    )

    parser.add_argument(
        '--val-interval',
        type=int,
        default=VAL_INTERVAL,
        help=f'每 N 个 epoch 进行一次验证 (默认: {VAL_INTERVAL})'
    )

    parser.add_argument(
        '--val-sample-size',
        type=int,
        default=VAL_SAMPLE_SIZE,
        help=f'每次验证随机抽取的实例数，0 表示用全部验证集 (默认: {VAL_SAMPLE_SIZE})'
    )

    parser.add_argument(
        '--val-sample-seed',
        type=int,
        default=VAL_SAMPLE_SEED,
        help=f'验证子集随机种子，固定以保证跨 epoch 指标可比 (默认: {VAL_SAMPLE_SEED})'
    )

    parser.add_argument(
        '--no-progress-limit',
        type=int,
        default=NO_PROGRESS_LIMIT,
        help=f'连续 N 步 env_timeline 无推进视为死锁并终止 episode (默认: {NO_PROGRESS_LIMIT})'
    )

    # === 早停参数 ===
    parser.add_argument(
        '--relative-tolerance',
        type=float,
        default=RELATIVE_TOLERANCE,
        help=f'早停相对容差：验证指标相对改善低于此值视为无显著改善 (默认: {RELATIVE_TOLERANCE})'
    )

    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=EARLY_STOP_PATIENCE,
        help=f'早停耐心度：连续无显著改善的验证 epoch 数 (默认: {EARLY_STOP_PATIENCE})'
    )

    parser.add_argument(
        '--min-epochs',
        type=int,
        default=MIN_EPOCHS,
        help=f'最小 epoch 数，低于此数不允许早停 (默认: {MIN_EPOCHS})'
    )

    # === 续跑与配置 ===
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

    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='从指定实验目录续跑（自动加载已完成的 epoch 和早停状态）'
    )

    # === 旧版收敛条件（仅 --legacy-convergence 时启用）===
    parser.add_argument(
        '--legacy-convergence',
        action='store_true',
        default=False,
        help='同时启用旧版绝对收敛条件（reward 阈值、loss 阈值等）'
    )

    parser.add_argument(
        '--reward-threshold',
        type=float,
        default=REWARD_THRESHOLD,
        help=f'[旧版] 累积 reward 阈值 (默认: {REWARD_THRESHOLD})'
    )

    parser.add_argument(
        '--loss-threshold',
        type=float,
        default=None,
        help='[旧版] Loss 阈值：当所有 loss 指标低于此值时判定收敛 (默认: 不检查)'
    )

    parser.add_argument(
        '--convergence-patience',
        type=int,
        default=CONVERGENCE_PATIENCE,
        help=f'[旧版] 收敛耐心度：连续满足所有收敛条件的迭代次数 (默认: {CONVERGENCE_PATIENCE})'
    )

    parser.add_argument(
        '--makespan-gap-threshold',
        type=float,
        default=MAKESPAN_GAP_THRESHOLD,
        help=f'[旧版] Makespan Gap 早停阈值 (默认: {MAKESPAN_GAP_THRESHOLD})'
    )

    parser.add_argument(
        '--reward-stability-window',
        type=int,
        default=None,
        help='[旧版] Reward 稳定性检测窗口 (默认: 不检查)'
    )

    parser.add_argument(
        '--reward-stability-threshold',
        type=float,
        default=REWARD_STABILITY_THRESHOLD,
        help=f'[旧版] Reward 变异系数阈值 (默认: {REWARD_STABILITY_THRESHOLD})'
    )

    parser.add_argument(
        '--epsilon-floor',
        type=float,
        default=None,
        help='[旧版] Epsilon 下界阈值 (默认: 不检查)'
    )

    # === 不确定性事件 ===
    parser.add_argument(
        '--uncertain-scenario',
        type=str,
        default=None,
        help='不确定性事件场景名（如 default, heavy），加载 dataset/uncertain-events/<scenario>/ 下的预生成事件数据'
    )

    return parser.parse_args()


def load_agent_config(agent_name: str) -> dict:
    """
    从 config/agents/ 目录加载指定 Agent 的配置文件（带缓存）

    Args:
        agent_name: Agent 名称（如 GraphPPOAgent, GraphDualAgent）

    Returns:
        dict: Agent 配置字典
    """
    global _cached_agent_config
    if agent_name in _cached_agent_config:
        return _cached_agent_config[agent_name]

    config_path = AGENTS_CONFIG_DIR / f"{agent_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Agent 配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)
    logger.info(f"已加载 Agent 配置: {config_path}")
    _cached_agent_config[agent_name] = agent_config
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

def generate_instance_config(parsed_data: dict, uncertain_events: Optional[List[dict]] = None) -> dict:
    """从解析的数据生成 job_config / map_config / event_config

    Args:
        parsed_data: parse_agv_instance() 的输出
        uncertain_events: 可选的预生成不确定性事件时间线（来自 dataset/uncertain-events/）
    """
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

    event_config = {
        "event_type": [
            "packet_factory.JUST_TEST",
            "packet_factory.ENV_PAUSED",
            "packet_factory.ENV_RECOVER",
            "packet_factory.ENV_RESTART",
            "packet_factory.AGV_FAIL",
            "packet_factory.MACHINE_FAIL",
            "packet_factory.JOB_ADD"
        ]
    }

    # 如果有预生成的不确定性事件，注入 event_timeline
    if uncertain_events:
        event_config["event_timeline"] = [{"event": e} for e in uncertain_events]

    return {
        "event_config": event_config,
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
    """构建完整的 bootstrap 配置（带缓存）

    从 config/agents/{agent_key}.yaml 加载 Agent 超参数，
    设置 ui_mode=backend, task_mode=training。
    基础 YAML 模板和 Agent 配置仅首次从磁盘读取，后续使用缓存。

    Args:
        agent_key: Agent 标识键
        instance_config: generate_instance_config() 的输出
        base_yaml_path: 可选的自定义基础 YAML 路径
    """
    global _cached_base_template

    # 缓存基础 YAML 模板（仅首次读取）
    if _cached_base_template is None or base_yaml_path is not None:
        yaml_path = base_yaml_path or str(DEFAULT_CONFIG_PATH)
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _cached_base_template = raw["config"]

    config = copy.deepcopy(_cached_base_template)

    # 从 per-agent 配置文件加载完整 agent 参数（超参数 + identity 字段，含 mode，带缓存）
    agent_full_config = load_agent_config(agent_key)

    # 填充 agent 段
    config["simulation"]["agent"].update(agent_full_config)

    # 训练模式：始终使用 backend + training
    config["simulation"]["ui_mode"] = "backend"
    config["simulation"]["task_mode"] = "training"

    # headless：禁用可视化器，避免每步 pygame 渲染 + PNG 编码 + clock.tick 帧率限速拖慢训练
    config["simulation"]["headless"] = True

    # 注入实例数据
    config["simulation"]["job_config"] = instance_config["job_config"]
    config["simulation"]["map_config"] = instance_config["map_config"]
    config["simulation"]["event_config"] = instance_config["event_config"]

    return config


# ==================== 不确定性事件加载 ====================

def load_uncertain_events_for_instance(instance_path: Path, scenario: str) -> Optional[List[dict]]:
    """
    加载与实例对应的不确定性事件数据

    根据实例文件的相对路径，在 dataset/uncertain-events/<scenario>/ 下查找对应的事件文件。

    Args:
        instance_path: AGV 实例文件路径（绝对路径，来自 DATA_DIR 的 glob 结果）
        scenario: 场景名称（如 'default', 'heavy'）

    Returns:
        event_timeline 列表，若不存在返回 None
    """
    from dataset import UNCERTAIN_EVENTS_DIR

    instance_path = Path(instance_path)

    # 解析相对路径
    try:
        relative = instance_path.relative_to(DATA_DIR)
    except ValueError:
        # 尝试将路径视为相对于 agv-instances 的路径
        relative = instance_path

    # 将 _agv.txt 后缀替换为 _agv_events.json
    event_filename = instance_path.stem + "_events.json"
    event_relative = relative.parent / event_filename

    event_path = Path(UNCERTAIN_EVENTS_DIR) / scenario / event_relative

    if not event_path.exists():
        return None

    with open(event_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("event_timeline", [])


# ==================== 数据集划分 ====================

def split_dataset(data_dir: Path, val_ratio: float = 0.2,
                  seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """将数据文件划分为训练集和验证集（按子目录分层采样）

    Args:
        data_dir: 数据文件根目录
        val_ratio: 验证集比例
        seed: 随机种子（确保可复现）

    Returns:
        (train_files, val_files): 训练集和验证集文件路径列表
    """
    all_files = sorted(data_dir.glob('**/*_agv.txt'))
    if not all_files:
        raise FileNotFoundError(f"未找到数据文件: {data_dir}/**/*_agv.txt")

    # 按子目录分组（分层采样，确保各子目录在训练集和验证集中都有代表）
    groups: Dict[str, List[Path]] = {}
    for f in all_files:
        group_key = str(f.parent.relative_to(data_dir))
        groups.setdefault(group_key, []).append(f)

    train_files = []
    val_files = []

    rng = random.Random(seed)

    for group_key, files in sorted(groups.items()):
        shuffled = files.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        # 如果组内文件数 <= 2，至少保留 1 个给训练集
        if len(shuffled) <= 2:
            n_val = 1
        val_files.extend(shuffled[:n_val])
        train_files.extend(shuffled[n_val:])

    logger.info(f"数据集划分完成（seed={seed}）: "
                f"{len(train_files)} 训练, {len(val_files)} 验证, "
                f"共 {len(all_files)} 个文件, {len(groups)} 个子目录")

    return train_files, val_files


# ==================== Agent 一次性初始化 ====================

def initialize_training(agent_key: str,
                        base_yaml_path: Optional[str] = None,
                        uncertain_scenario: Optional[str] = None) -> object:
    """一次性初始化：加载配置、扫描组件、创建 Agent

    与 bootstrap() 不同，此函数只创建 Agent 而不创建 env。
    Agent 将跨 episode 复用，env 在每个 episode 独立创建。

    Args:
        agent_key: Agent 类名
        base_yaml_path: 可选的自定义基础 YAML 路径
        uncertain_scenario: 可选的不确定性事件场景名

    Returns:
        Agent 实例（持久化）
    """
    from executor.packet_factory.registry import load_config, scan_and_register_components
    from executor.packet_factory.lifecycle.initializer.agent_initializer import initialize_agent

    # 加载 Agent 配置
    agent_config = load_agent_config(agent_key)

    # 构建初始配置（使用第一个可用的数据文件，仅用于满足 load_config 的要求）
    data_files = list(DATA_DIR.glob('**/*_agv.txt'))
    if not data_files:
        raise FileNotFoundError(f"未找到数据文件: {DATA_DIR}")

    dummy_file = data_files[0]
    parsed_data = parse_agv_instance(dummy_file)
    uncertain_events = None
    if uncertain_scenario:
        uncertain_events = load_uncertain_events_for_instance(dummy_file, uncertain_scenario)
        if uncertain_events:
            logger.info(f"已加载不确定性事件: 场景={uncertain_scenario}, {len(uncertain_events)} 个事件")
    instance_config = generate_instance_config(parsed_data, uncertain_events=uncertain_events)
    config = build_config(agent_key, instance_config, base_yaml_path=base_yaml_path)

    # 存储配置到全局注册表
    load_config(config)

    # 一次性扫描注册组件
    scan_and_register_components()

    # 创建 Agent（env 将在后续每个 episode 单独创建）
    agent = initialize_agent(config)

    logger.info(f"Agent 初始化完成: {agent_key} (task_mode={agent.task_mode})")
    return agent


# ==================== Per-episode 环境创建 ====================

def create_env_for_instance(agent: object, instance_config: dict,
                            agent_key: str,
                            base_yaml_path: Optional[str] = None) -> object:
    """为特定实例创建新的 env，复用已有的 Agent

    每次调用会创建一个全新的 env（不同实例有不同的工厂布局），
    但 Agent 是同一个实例（权重持久化）。

    Args:
        agent: 已有的 Agent 实例
        instance_config: generate_instance_config() 的输出
        agent_key: Agent 类名
        base_yaml_path: 可选的自定义基础 YAML 路径

    Returns:
        新创建的环境实例
    """
    from executor.packet_factory.registry import load_config
    from executor.packet_factory.lifecycle.initializer.env_initializer import initialize_env
    from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import EnvStatus

    # 构建新实例的配置
    config = build_config(agent_key, instance_config, base_yaml_path=base_yaml_path)

    # 更新全局注册表中的配置
    load_config(config)

    # 创建新环境（复用已有 Agent）
    env = initialize_env(config, agent)

    # 设置 Agent 对环境的引用
    agent.context = env

    # Headless 模式：config 已设置 headless=True，initialize_env 不会注册可视化器，
    # 因此 env.env_visualizer 始终为 None（不会在 reset()/refresh_status() 中被重新安装）
    env.status = EnvStatus.RUNNING

    # 重置环境（加载实例数据，调用 agent.new_episode()，重置 agent.alive=True）
    env.reset()

    return env


# ==================== Episode 运行 ====================

def run_episode_with_env(agent: object, env: object,
                         train_interval: int = TRAIN_INTERVAL,
                         timeout: int = EPISODE_TIMEOUT,
                         is_training: bool = True,
                         no_progress_limit: int = NO_PROGRESS_LIMIT) -> dict:
    """使用已有的 agent+env 运行一次完整 episode

    与旧版 run_training_episode() 不同，此函数不调用 bootstrap()，
    Agent 和环境已经存在。

    Args:
        agent: Agent 实例（持久化）
        env: 环境实例（为当前数据文件创建）
        train_interval: 训练间隔步数
        timeout: 单次 episode 超时秒数
        is_training: 是否为训练模式（False 时切换到推理模式）

    Returns:
        dict: 包含 status, makespan, metrics, steps, elapsed 等字段
    """
    from executor.packet_factory.packet_factory.Agent.BaseAgent import INFERENCE

    # 验证模式：临时切换 task_mode
    original_task_mode = None
    original_mode = None
    if not is_training:
        original_task_mode = agent.task_mode
        original_mode = agent.mode
        agent.task_mode = INFERENCE
        agent.mode = f"{agent.ui_mode}_{INFERENCE}"

    start_time = time.time()
    step_count = 0
    # 死锁检测：env_timeline 连续无推进则提前终止，避免推理策略把实例推入
    # 无法完成的死局后空转到 wall-clock 超时（验证默认 600s、训练默认 7200s）
    last_timeline = env.env_timeline
    no_progress_steps = 0

    try:
        while not env.env_is_finished():
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

            # 死锁检测：timeline 未推进则累计，连续超过阈值即视为死局止损
            if env.env_timeline == last_timeline:
                no_progress_steps += 1
                if no_progress_steps >= no_progress_limit:
                    logger.warning(
                        f"Episode 死锁（连续 {no_progress_steps} 步 timeline 无推进，"
                        f"timeline={env.env_timeline}，已执行 {step_count} 步）"
                    )
                    metrics = agent.get_training_metrics() if hasattr(agent, 'get_training_metrics') else {}
                    return {
                        "status": "stuck",
                        "makespan": env.env_timeline,
                        "metrics": metrics,
                        "decision_stats": agent.get_decision_stats() if hasattr(agent, 'get_decision_stats') else {},
                        "steps": step_count,
                        "elapsed": time.time() - start_time,
                    }
            else:
                last_timeline = env.env_timeline
                no_progress_steps = 0

            # 训练更新（仅训练模式，每 train_interval 步）
            if is_training and step_count % train_interval == 0:
                if hasattr(agent, 'update'):
                    agent.update(observations, rewards)
                elif hasattr(agent, 'train'):
                    agent.train(observations, rewards, terminations, truncations, infos)

            # 每 50 步打印训练指标（仅训练模式）
            if is_training and step_count % 50 == 0 and hasattr(agent, 'get_training_metrics'):
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

    except KeyboardInterrupt:
        # Ctrl+C：立即停止当前 episode，向上抛出
        logger.info("Episode 被键盘中断")
        raise
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
    finally:
        # 恢复原始 task_mode
        if not is_training and original_task_mode is not None:
            agent.task_mode = original_task_mode
            agent.mode = original_mode


# ==================== Makespan 下界计算 ====================

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


# ==================== Epoch 级早停检测 ====================

def should_early_stop(early_stopping_state: dict, current_val_gap: float,
                       tolerance: float, patience: int, min_epochs: int,
                       current_epoch: int) -> Tuple[bool, str]:
    """基于增量跟踪的早停判断

    此函数更新 early_stopping_state 并判断是否应早停。
    早停状态在调用间持久化，支持正确的增量判断。

    Args:
        early_stopping_state: 早停状态字典（含 best_val_gap, best_epoch, patience_counter）
        current_val_gap: 当前 epoch 的验证 makespan gap
        tolerance: 相对容差阈值
        patience: 早停耐心度
        min_epochs: 最小 epoch 数
        current_epoch: 当前 epoch 编号

    Returns:
        Tuple[bool, str]: (是否应早停, 原因描述)
    """
    best_val_gap = early_stopping_state['best_val_gap']

    if current_val_gap < best_val_gap:
        relative_improvement = (best_val_gap - current_val_gap) / max(abs(best_val_gap), 1e-8)
        if relative_improvement > tolerance:
            # 显著改善 → 重置耐心度
            early_stopping_state['patience_counter'] = 0
        else:
            # 边际改善 → 更新最佳，但增加耐心度
            early_stopping_state['patience_counter'] += 1
        early_stopping_state['best_val_gap'] = current_val_gap
        early_stopping_state['best_epoch'] = current_epoch
    else:
        # 无改善 → 增加耐心度
        early_stopping_state['patience_counter'] += 1

    # 判断早停
    if (early_stopping_state['patience_counter'] >= patience
            and current_epoch >= min_epochs):
        return True, (f"验证集 makespan gap 连续 {patience} 个 epoch "
                      f"无显著改善 (相对容差 {tolerance:.1%}), "
                      f"当前={current_val_gap:.4f}, "
                      f"最佳={early_stopping_state['best_val_gap']:.4f} "
                      f"(epoch {early_stopping_state['best_epoch']})")

    return False, ""


def check_convergence(iteration_metrics: List[Dict], reward_threshold: float,
                      loss_threshold: Optional[float], patience: int,
                      min_iters: int = 30, current_iteration: int = 0,
                      makespan_gap_threshold: float = 0.30,
                      reward_stability_window: Optional[int] = None,
                      reward_stability_threshold: float = 0.05,
                      epsilon_floor: Optional[float] = None) -> Tuple[bool, str]:
    """检查训练是否收敛（旧版，含 min_iters 保护 + Makespan Gap 早停 + Reward 稳定性 + Epsilon 下界）

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


# ==================== Epoch 指标计算 ====================

def compute_epoch_metrics(train_results: List[dict], val_results: List[dict],
                          epoch: int, agent: object = None) -> dict:
    """计算单个 epoch 的聚合指标

    Args:
        train_results: 训练 episode 结果列表
        val_results: 验证 episode 结果列表
        epoch: 当前 epoch 编号
        agent: Agent 实例（用于获取 epsilon 等指标）

    Returns:
        dict: 包含训练和验证的聚合指标
    """
    # 训练指标
    train_rewards = [r.get('metrics', {}).get('episode_reward', 0.0)
                     for r in train_results if r.get('status') == 'completed']
    train_gaps = [r.get('makespan_gap', float('inf'))
                  for r in train_results if r.get('status') == 'completed']
    train_steps = [r.get('steps', 0) for r in train_results]
    train_elapsed = [r.get('elapsed', 0.0) for r in train_results]

    # 验证指标
    val_gaps = [r.get('makespan_gap', float('inf'))
                for r in val_results if r.get('status') == 'completed']
    val_makespans = [r.get('makespan', 0.0)
                     for r in val_results if r.get('status') == 'completed']

    # 如果没有验证结果，使用 inf 表示缺失
    if not val_gaps:
        val_mean_gap = float('inf')
        val_mean_makespan = 0.0
    else:
        val_mean_gap = sum(val_gaps) / len(val_gaps)
        val_mean_makespan = sum(val_makespans) / max(len(val_makespans), 1)

    # Agent 指标
    epsilon = None
    if agent is not None and hasattr(agent, 'epsilon'):
        epsilon = agent.epsilon

    loss_metrics = {}
    if agent is not None and hasattr(agent, 'get_training_metrics'):
        agent_metrics = agent.get_training_metrics()
        loss_metrics = {k: v for k, v in agent_metrics.items()
                        if 'loss' in k.lower() and isinstance(v, (int, float))}

    n_train_completed = len(train_rewards)
    n_val_completed = len(val_gaps)
    n_train_total = len(train_results)
    n_val_total = len(val_results)

    result = {
        'epoch': epoch,
        # 训练指标
        'train_mean_reward': sum(train_rewards) / max(len(train_rewards), 1),
        'train_mean_makespan_gap': sum(train_gaps) / max(len(train_gaps), 1),
        'train_mean_steps': sum(train_steps) / max(len(train_steps), 1),
        'train_total_elapsed': sum(train_elapsed),
        'train_completed': n_train_completed,
        'train_total': n_train_total,
        # 验证指标
        'val_makespan_gap': val_mean_gap,
        'val_mean_makespan': val_mean_makespan,
        'val_completed': n_val_completed,
        'val_total': n_val_total,
        # Agent 指标
        'epsilon': epsilon,
        'loss_metrics': loss_metrics,
    }

    return result


# ==================== 结果管理 ====================

def save_epoch_result(experiment_dir: Path, record: dict):
    """追加一条 epoch 结果到 JSONL 文件

    Args:
        experiment_dir: 实验目录
        record: epoch 结果字典
    """
    results_file = experiment_dir / "epoch_results.jsonl"
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_epochs(experiment_dir: Path) -> Tuple[int, dict]:
    """读取已完成的 epoch 数和早停状态

    Args:
        experiment_dir: 实验目录

    Returns:
        (completed_epochs, early_stopping_state): 已完成的 epoch 数和早停状态字典
    """
    results_file = experiment_dir / "epoch_results.jsonl"
    if not results_file.exists():
        return 0, {'best_val_gap': float('inf'), 'best_epoch': -1, 'patience_counter': 0}

    count = 0
    best_val_gap = float('inf')
    best_epoch = -1
    patience_counter = 0

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                count += 1
                val_gap = record.get('val_makespan_gap')
                if val_gap is not None and val_gap < best_val_gap:
                    best_val_gap = val_gap
                    best_epoch = record.get('epoch', count - 1)
                # 重建 patience_counter
                patience_counter = record.get('patience_counter', 0)
            except json.JSONDecodeError:
                continue

    return count, {
        'best_val_gap': best_val_gap,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
    }


def save_best_checkpoint(agent: object, experiment_dir: Path,
                         epoch: int, epoch_metrics: dict):
    """保存最佳模型检查点

    Args:
        agent: Agent 实例
        experiment_dir: 实验目录
        epoch: 当前 epoch 编号
        epoch_metrics: 当前 epoch 指标
    """
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    if hasattr(agent, 'save_model'):
        try:
            model_path = agent.save_model()
            if model_path:
                logger.info(f"最佳模型已保存到: {model_path}")
        except Exception as e:
            logger.warning(f"保存模型失败: {e}")

    # 保存检查点元数据
    meta = {
        'epoch': epoch,
        'val_makespan_gap': epoch_metrics.get('val_makespan_gap'),
        'train_mean_reward': epoch_metrics.get('train_mean_reward'),
        'train_mean_makespan_gap': epoch_metrics.get('train_mean_makespan_gap'),
        'epsilon': epoch_metrics.get('epsilon'),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = checkpoint_dir / "best_checkpoint_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"检查点元数据已保存: {meta_path} (epoch={epoch}, "
                f"val_gap={epoch_metrics.get('val_makespan_gap', 'N/A'):.4f})")


def save_training_result(agent, env, result: dict, experiment_dir: Path):
    """保存训练结果

    复刻 BackendCore._save_training_results() 的逻辑，
    保存到实验目录下的 training_report.json。

    Args:
        agent: Agent 实例
        env: 环境实例
        result: run_episode_with_env() 的返回值
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


# ==================== 主训练循环（Epoch 级别）====================

def main():
    """主函数"""
    args = parse_args()

    # 设置日志级别
    setup_logging(args.log_level, args.backend_log_level)

    # 确定实验 ID 和目录
    if args.resume_from:
        # 续跑模式：使用已有实验目录
        experiment_dir = Path(args.resume_from)
        if not experiment_dir.exists():
            logger.error(f"续跑目录不存在: {experiment_dir}")
            return
        # 尝试读取已有实验 ID
        config_file = experiment_dir / "experiment_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            experiment_id = existing_config.get('experiment_id', experiment_dir.name)
        else:
            experiment_id = experiment_dir.name
    else:
        experiment_id = args.experiment_id or time.strftime("train_%Y%m%d_%H%M%S")
        experiment_dir = RESULTS_DIR / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

    # 划分数据集
    train_files, val_files = split_dataset(DATA_DIR, args.val_ratio, args.split_seed)

    # 验证子集：固定随机抽样一份子集，跨 epoch 复用 → 早停指标可比且单次验证耗时可控
    val_files_full = val_files
    if args.val_sample_size > 0 and len(val_files) > args.val_sample_size:
        rng = random.Random(args.val_sample_seed)
        val_files = rng.sample(val_files, args.val_sample_size)
        logger.info(f"验证子集: 从 {len(val_files_full)} 个验证实例中固定抽取 {len(val_files)} 个 "
                    f"(seed={args.val_sample_seed})，跨 epoch 复用")
    else:
        val_files_full = val_files

    logger.info("=" * 60)
    logger.info("自动化训练循环脚本（Epoch 级别，Mini-Batch 训练）")
    logger.info(f"训练 Agent: {args.agent}")
    logger.info(f"Epoch 大小: {args.epoch_size} 个训练 episode/epoch")
    logger.info(f"最大 Epoch 数: {args.max_epochs}")
    logger.info(f"最小 Epoch 数: {args.min_epochs}")
    logger.info(f"验证集比例: {args.val_ratio:.0%}")
    logger.info(f"验证间隔: 每 {args.val_interval} 个 epoch")
    logger.info(f"早停相对容差: {args.relative_tolerance:.1%}")
    logger.info(f"早停耐心度: {args.early_stop_patience}")
    logger.info(f"训练间隔: 每 {args.train_interval} 步")
    logger.info(f"训练 episode 超时: {args.episode_timeout}s，验证 episode 超时: {args.val_episode_timeout}s")
    logger.info(f"死锁检测: 连续 {args.no_progress_limit} 步 timeline 无推进即终止 episode")
    logger.info(f"数据集: {len(train_files)} 训练, {len(val_files)} 验证"
                f"{'（子集，全集 ' + str(len(val_files_full)) + '）' if val_files is not val_files_full else ''}")
    if args.legacy_convergence:
        logger.info(f"旧版收敛: reward阈值={args.reward_threshold}, "
                    f"loss阈值={args.loss_threshold}, "
                    f"patience={args.convergence_patience}")
    logger.info(f"实验目录: {experiment_dir}")
    logger.info("=" * 60)

    # 保存实验配置快照
    config_snapshot = {
        "experiment_id": experiment_id,
        "agent": args.agent,
        "epoch_size": args.epoch_size,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "val_interval": args.val_interval,
        "relative_tolerance": args.relative_tolerance,
        "early_stop_patience": args.early_stop_patience,
        "train_interval": args.train_interval,
        "episode_timeout": args.episode_timeout,
        "val_episode_timeout": args.val_episode_timeout,
        "no_progress_limit": args.no_progress_limit,
        "val_sample_size": args.val_sample_size,
        "val_sample_seed": args.val_sample_seed,
        "config_yaml": args.config_yaml,
        "legacy_convergence": args.legacy_convergence,
        "reward_threshold": args.reward_threshold if args.legacy_convergence else None,
        "loss_threshold": args.loss_threshold if args.legacy_convergence else None,
        "convergence_patience": args.convergence_patience if args.legacy_convergence else None,
        "train_files": [str(f.relative_to(DATA_DIR)) for f in train_files],
        "val_files": [str(f.relative_to(DATA_DIR)) for f in val_files],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(experiment_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    # 不注册自定义 SIGINT 处理器 —— 让 KeyboardInterrupt 自然抛出
    # 这样 Ctrl+C 可以立即中断当前操作（包括 PyTorch 训练步）
    # 在 epoch 级别用 try/except KeyboardInterrupt 捕获并优雅保存

    # === 一次性 Agent 初始化 ===
    agent = initialize_training(args.agent, base_yaml_path=args.config_yaml,
                               uncertain_scenario=args.uncertain_scenario)

    # === 加载已完成的 epoch（续跑支持）===
    start_epoch, early_stopping_state = load_completed_epochs(experiment_dir)
    if start_epoch > 0:
        logger.info(f"检测到已完成的 {start_epoch} 个 epoch，将从第 {start_epoch} 个 epoch 继续")
        logger.info(f"早停状态: best_val_gap={early_stopping_state['best_val_gap']:.4f}, "
                    f"best_epoch={early_stopping_state['best_epoch']}, "
                    f"patience_counter={early_stopping_state['patience_counter']}")

    # 跨 epoch 指标追踪
    epoch_metrics_list: List[Dict] = []

    # 旧版收敛指标（仅 legacy 模式）
    iteration_metrics: List[Dict] = []

    try:
        for epoch in range(start_epoch, args.max_epochs):
            epoch_start_time = time.time()

            # ========== 训练阶段（Mini-Batch）==========
            epoch_train_files = random.sample(
                train_files,
                min(args.epoch_size, len(train_files))
            )
            train_results = []

            logger.info(f"\n{'=' * 60}")
            logger.info(f"[Epoch {epoch}] 训练阶段 ({len(epoch_train_files)} episodes)")
            logger.info(f"{'=' * 60}")

            for ep_idx, data_file in enumerate(epoch_train_files):
                relative_path = data_file.relative_to(DATA_DIR)

                try:
                    parsed_data = parse_agv_instance(data_file)
                    lower_bound = compute_makespan_lower_bound(parsed_data)
                except Exception as e:
                    logger.error(f"解析数据文件失败 {relative_path}: {e}")
                    train_results.append({
                        'status': 'error', 'makespan': 0.0,
                        'metrics': {}, 'steps': 0, 'elapsed': 0.0,
                        'makespan_gap': float('inf'),
                        'data_file': str(relative_path),
                    })
                    continue

                instance_config = generate_instance_config(parsed_data)

                # 加载不确定性事件（如果指定了场景）
                if args.uncertain_scenario:
                    uncertain_events = load_uncertain_events_for_instance(data_file, args.uncertain_scenario)
                    if uncertain_events:
                        instance_config = generate_instance_config(parsed_data, uncertain_events=uncertain_events)

                # 创建环境（复用 Agent）
                try:
                    env = create_env_for_instance(
                        agent, instance_config, args.agent,
                        base_yaml_path=args.config_yaml
                    )
                except Exception as e:
                    logger.error(f"创建环境失败 {relative_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    train_results.append({
                        'status': 'error', 'makespan': 0.0,
                        'metrics': {}, 'steps': 0, 'elapsed': 0.0,
                        'makespan_gap': float('inf'),
                        'data_file': str(relative_path),
                    })
                    continue

                # 运行训练 episode
                try:
                    result = run_episode_with_env(
                        agent, env,
                        train_interval=args.train_interval,
                        timeout=args.episode_timeout,
                        is_training=True,
                        no_progress_limit=args.no_progress_limit,
                    )
                except KeyboardInterrupt:
                    logger.info("\n训练 episode 被中断，保存当前 epoch 结果...")
                    raise
                result['makespan_gap'] = (
                    (result.get('makespan', 0.0) - lower_bound) / lower_bound
                    if lower_bound > 0 else float('inf')
                )
                result['makespan_lower_bound'] = lower_bound
                result['data_file'] = str(relative_path)
                train_results.append(result)

                # 释放本 episode 的 env（同验证分支，避免循环引用导致的内存累积）
                del env

                # 打印本次 episode 的关键指标
                ep_reward = result.get('metrics', {}).get('episode_reward', 0.0)
                makespan = result.get('makespan', 0.0)
                gap = result.get('makespan_gap', float('inf'))
                status = result.get('status', 'unknown')

                if status == 'completed':
                    logger.info(f"  [Epoch {epoch} / Episode {ep_idx}] "
                                f"文件={relative_path}, "
                                f"makespan={makespan:.2f}, "
                                f"gap={gap:.1%}, "
                                f"reward={ep_reward:.4f}, "
                                f"steps={result.get('steps', 0)}, "
                                f"elapsed={result.get('elapsed', 0.0):.2f}s")
                else:
                    logger.warning(f"  [Epoch {epoch} / Episode {ep_idx}] "
                                   f"文件={relative_path}, "
                                   f"状态={status}, makespan={makespan:.2f}")


            # ========== 验证阶段 ==========
            val_results = []
            do_validation = (epoch % args.val_interval == 0)

            if do_validation and val_files:
                logger.info(f"\n[Epoch {epoch}] 验证阶段 ({len(val_files)} episodes)")

                for data_file in val_files:
                    relative_path = data_file.relative_to(DATA_DIR)

                    try:
                        parsed_data = parse_agv_instance(data_file)
                        lower_bound = compute_makespan_lower_bound(parsed_data)
                    except Exception as e:
                        logger.error(f"解析验证文件失败 {relative_path}: {e}")
                        continue

                    instance_config = generate_instance_config(parsed_data)

                    # 加载不确定性事件（如果指定了场景）
                    if args.uncertain_scenario:
                        uncertain_events = load_uncertain_events_for_instance(data_file, args.uncertain_scenario)
                        if uncertain_events:
                            instance_config = generate_instance_config(parsed_data, uncertain_events=uncertain_events)

                    try:
                        env = create_env_for_instance(
                            agent, instance_config, args.agent,
                            base_yaml_path=args.config_yaml
                        )
                    except Exception as e:
                        logger.error(f"创建验证环境失败 {relative_path}: {e}")
                        continue

                    # 验证模式：推理（不训练）；用更短的验证超时 + 死锁检测控制单实例成本
                    try:
                        result = run_episode_with_env(
                            agent, env,
                            timeout=args.val_episode_timeout,
                            is_training=False,
                            no_progress_limit=args.no_progress_limit,
                        )
                    except KeyboardInterrupt:
                        logger.info("\n验证 episode 被中断，保存当前 epoch 结果...")
                        raise
                    result['makespan_gap'] = (
                        (result.get('makespan', 0.0) - lower_bound) / lower_bound
                        if lower_bound > 0 else float('inf')
                    )
                    result['makespan_lower_bound'] = lower_bound
                    result['data_file'] = str(relative_path)
                    val_results.append(result)

                    # 验证每 episode 输出指标（与训练分支对齐，否则验证阶段只能看到 [Callback] 行）
                    v_status = result.get('status', 'unknown')
                    v_makespan = result.get('makespan', 0.0)
                    v_gap = result.get('makespan_gap', float('inf'))
                    v_steps = result.get('steps', 0)
                    v_elapsed = result.get('elapsed', 0.0)
                    logger.info(f"  [Epoch {epoch} / 验证 {len(val_results)}/{len(val_files)}] "
                                f"文件={relative_path}, 状态={v_status}, "
                                f"makespan={v_makespan:.2f}, gap={v_gap:.1%}, "
                                f"steps={v_steps}, elapsed={v_elapsed:.2f}s")

                    # 释放本 episode 的 env（env↔event_queue↔agent.context 存在循环引用，
                    # 显式断开 + 触发回收，避免单 epoch 累积数十个 env 导致内存持续上涨）
                    del env

            # ========== 计算 Epoch 指标 ==========
            epoch_metrics = compute_epoch_metrics(
                train_results, val_results, epoch, agent=agent
            )

            # ========== 早停检测 ==========
            # 仅在真正执行了验证的 epoch 更新早停状态。非验证 epoch 的
            # val_makespan_gap 为 inf，若参与计数会把"跳过验证"误判为"无改善"，
            # 在 val_interval>1 时导致 patience 被 4 个跳过 epoch 空耗 → 假早停。
            # 因此 patience 现在按"验证 epoch"计数（而非所有 epoch）。
            did_validate = bool(val_results)
            if did_validate:
                current_val_gap = epoch_metrics.get('val_makespan_gap', float('inf'))
                best_val_gap = early_stopping_state['best_val_gap']

                stop, reason = should_early_stop(
                    early_stopping_state, current_val_gap,
                    tolerance=args.relative_tolerance,
                    patience=args.early_stop_patience,
                    min_epochs=args.min_epochs,
                    current_epoch=epoch,
                )

                # 日志输出
                if current_val_gap < best_val_gap:
                    relative_improvement = (best_val_gap - current_val_gap) / max(abs(best_val_gap), 1e-8)
                    if relative_improvement > args.relative_tolerance:
                        logger.info(f"  验证指标显著改善: {best_val_gap:.4f} → {current_val_gap:.4f} "
                                    f"(改善 {relative_improvement:.1%})")
                    else:
                        logger.info(f"  验证指标边际改善: {best_val_gap:.4f} → {current_val_gap:.4f} "
                                    f"(改善 {relative_improvement:.1%} < 容差 {args.relative_tolerance:.1%})")
                    # 保存最佳模型检查点
                    save_best_checkpoint(agent, experiment_dir, epoch, epoch_metrics)
                else:
                    logger.info(f"  验证指标无改善: {current_val_gap:.4f} >= 最佳 {best_val_gap:.4f} "
                                f"(耐心度 {early_stopping_state['patience_counter']}"
                                f"/{args.early_stop_patience})")
            else:
                current_val_gap = epoch_metrics.get('val_makespan_gap', float('inf'))
                stop, reason = False, ""
                logger.info(f"  本 epoch 跳过验证（val_interval={args.val_interval}），"
                            f"早停状态保持 (耐心度 {early_stopping_state['patience_counter']}"
                            f"/{args.early_stop_patience})")

            epoch_metrics['best_val_gap'] = early_stopping_state['best_val_gap']
            epoch_metrics['best_epoch'] = early_stopping_state['best_epoch']
            epoch_metrics['patience_counter'] = early_stopping_state['patience_counter']
            epoch_metrics_list.append(epoch_metrics)

            # 早停判断
            if stop:
                logger.info(f"\n=== 早停触发 ===")
                logger.info(f"原因: {reason}")
                logger.info(f"最佳验证 makespan gap: {early_stopping_state['best_val_gap']:.4f} "
                            f"(epoch {early_stopping_state['best_epoch']})")
                # 保存当前 epoch 结果后退出
                epoch_elapsed = time.time() - epoch_start_time
                epoch_metrics['epoch_elapsed'] = epoch_elapsed
                save_epoch_result(experiment_dir, epoch_metrics)
                break

            # ========== 旧版收敛检测（可选）==========
            if args.legacy_convergence:
                # 将本次 epoch 的训练指标加入旧版检测列表
                epoch_train_metrics = epoch_metrics.copy()
                epoch_train_metrics['episode_reward'] = epoch_metrics.get('train_mean_reward', 0.0)
                iteration_metrics.append(epoch_train_metrics)

                if epoch > 0 and iteration_metrics:
                    converged, reason = check_convergence(
                        iteration_metrics, args.reward_threshold, args.loss_threshold,
                        args.convergence_patience,
                        min_iters=args.min_epochs, current_iteration=epoch,
                        makespan_gap_threshold=args.makespan_gap_threshold,
                        reward_stability_window=args.reward_stability_window,
                        reward_stability_threshold=args.reward_stability_threshold,
                        epsilon_floor=args.epsilon_floor,
                    )
                    if converged:
                        logger.info(f"\n=== 旧版收敛条件触发 ===")
                        logger.info(f"收敛原因: {reason}")
                        epoch_elapsed = time.time() - epoch_start_time
                        epoch_metrics['epoch_elapsed'] = epoch_elapsed
                        save_epoch_result(experiment_dir, epoch_metrics)
                        break
                    elif reason:
                        logger.info(f"旧版收敛未满足: {reason}")

            # ========== 保存 Epoch 结果 ==========
            epoch_elapsed = time.time() - epoch_start_time
            epoch_metrics['epoch_elapsed'] = epoch_elapsed
            save_epoch_result(experiment_dir, epoch_metrics)

            # ========== 打印 Epoch 总结 ==========
            train_mean_reward = epoch_metrics.get('train_mean_reward', 0.0)
            train_mean_gap = epoch_metrics.get('train_mean_makespan_gap', float('inf'))
            val_mean_gap = epoch_metrics.get('val_makespan_gap', float('inf'))
            epsilon = epoch_metrics.get('epsilon', 'N/A')

            logger.info(f"\n[Epoch {epoch}] 完成! "
                        f"训练 reward={train_mean_reward:.4f}, "
                        f"训练 gap={train_mean_gap:.1%}, "
                        f"验证 gap={val_mean_gap:.1%}, "
                        f"最佳验证 gap={early_stopping_state['best_val_gap']:.4f}, "
                        f"epsilon={epsilon}, "
                        f"耗时={epoch_elapsed:.1f}s")

            # Epoch 边界回收本 epoch 累积的 env 循环引用（del env 断开强引用后触发分代回收）
            gc.collect()

        # ========== 最终统计 ==========
        if epoch_metrics_list:
            final = epoch_metrics_list[-1]
            best = early_stopping_state

            logger.info("\n" + "=" * 60)
            logger.info("训练完成!")
            logger.info(f"总 Epoch 数: {len(epoch_metrics_list)}")
            logger.info(f"最终训练 reward: {final.get('train_mean_reward', 0.0):.4f}")
            logger.info(f"最终训练 makespan gap: {final.get('train_mean_makespan_gap', float('inf')):.1%}")
            logger.info(f"最终验证 makespan gap: {final.get('val_makespan_gap', float('inf')):.1%}")
            logger.info(f"最佳验证 makespan gap: {best['best_val_gap']:.4f} "
                        f"(epoch {best['best_epoch']})")
            logger.info(f"结果保存在: {experiment_dir}")
            logger.info("=" * 60)
        else:
            logger.info("未完成任何 epoch")

    except KeyboardInterrupt:
        logger.info("\n收到键盘中断信号，正在保存当前进度...")
        # 保存当前 epoch 的部分结果（如果有）
        if epoch_metrics_list:
            save_epoch_result(experiment_dir, epoch_metrics_list[-1])
        logger.info("进度已保存")
    finally:
        logger.info("训练循环已退出")


if __name__ == "__main__":
    main()
