"""
自动化调度算法对比实验脚本

功能：
1. 直接调用 executor 层运行 DualDRLAgent / ORToolsAgent / ORToolsBatchAgent
2. 在 agv-instances 各数据集族上运行实验
3. 实时保存结果到 JSONL（支持中断续跑）
4. 自动构建配置（支持 --config-yaml 覆盖基础模板）
5. 多进程/多线程并行运行，充分利用多核 CPU（-j/--workers、--parallel-mode）

用法：
    uv run python scripts/run_benchmark.py --agents ORToolsAgent --families kacem --runs-opt 1
    uv run python scripts/run_benchmark.py --agents DualDRLAgent ORToolsAgent ORToolsBatchAgent
    uv run python scripts/run_benchmark.py --config-yaml custom.yaml --families brandimarte
    uv run python scripts/run_benchmark.py --log-level DEBUG --backend-log-level INFO
    uv run python scripts/run_benchmark.py --small-only                           # 只运行小规模数据集
    uv run python scripts/run_benchmark.py --small-only --small-max-jobs 10       # 自定义小规模阈值
    uv run python scripts/run_benchmark.py -j 8                                   # 8 进程并行
    uv run python scripts/run_benchmark.py --parallel-mode thread -j 8            # 多线程模式
    uv run python scripts/run_benchmark.py --agents DualDRLAgent --device cpu      # 强制 DRL agent 用 CPU（覆盖 YAML device: cuda）

中断后续跑：
    再次运行相同命令即可，已完成的 trial 会自动跳过
    （status=error 的记录会被删除并重跑）

日志级别控制：
    --log-level: 控制前端脚本的日志输出级别 (默认: INFO)
    --backend-log-level: 控制后端 logger 的日志输出级别 (默认: WARNING)

并行控制：
    -j/--workers: 并行 worker 数 (默认 CPU 核数)
    --parallel-mode: process(默认,多进程真多核) | thread(多线程,受 GIL 限制)
    --ortools-workers: OR-Tools 每 solver 内部线程数 (默认 auto=cpu_count//workers, 避免 4×N 超额订阅)
    注意: DRL agent 使用 device: cuda 时，多进程会各自创建 CUDA context，
          显存吃紧可降低 -j 或在 agent yaml 中设置 device: cpu。
"""

import argparse
import copy
import json
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# === 路径常量 ===
DATA_DIR = PROJECT_ROOT / "dataset" / "agv-instances"
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"
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

# === 日志 ===
logger = logging.getLogger("benchmark")

# === Agent 配置映射 ===
# mode 由 config/agents/ 下的 YAML 文件提供
# ui_mode / task_mode 由 application_config.yaml 全局控制
# 此处仅保留 benchmark 专用的覆盖参数
AGENT_CONFIGS = {
    "DualDRLAgent": {
        "model_path": "./training_logs/models/DualDRLAgent/agent_model.pt",
    },
    "ORToolsAgent": {
        "time_limit_seconds": 30,
    },
    "ORToolsBatchAgent": {
        "time_limit_seconds": 60,
    },
    "GraphDPAgent": {
        "model_path": "./training_logs/models/GraphDPAgent/agent_model.pt",
    },
    "GraphDualAgent": {
        "model_path": "./training_logs/models/GraphDualAgent/agent_model.pt",
    },
    "GraphPPOAgent": {
        "model_path": "./training_logs/models/GraphPPOAgent/agent_model.pt",
    },
    "GraphGRPOAgent": {
        "model_path": "./training_logs/models/GraphGRPOAgent/agent_model.pt",
    },
}

ALL_FAMILIES = ["barnes", "behnke", "brandimarte", "dauzere", "fattahi", "hurink", "kacem"]

# DRL agents use runs_drl; optimization agents use runs_opt
DRL_AGENTS = {
    "DualDRLAgent", "GraphDPAgent", "GraphDualAgent",
    "GraphPPOAgent", "GraphGRPOAgent",
}

# OR-Tools 优化 agent 集合：这些 agent 的每次 solve 会启动 num_workers 个内部线程，
# 并行 benchmark 时需要按 cpu_count // workers 调整 num_workers 以避免超额订阅 CPU。
OR_TOOLS_AGENTS = {"ORToolsAgent", "ORToolsBatchAgent"}

# === 全局中断标记 ===
_shutdown_requested = False
# 进程模式下由 _init_worker 在每个 worker 进程设置；线程模式下主进程共享（置 None 即不检查）。
_worker_shutdown_event = None


def setup_logging(log_level: str = "INFO", backend_log_level: str = "WARNING"):
    """设置前端和后端日志级别
    
    Args:
        log_level: 前端脚本日志级别 (DEBUG, INFO, WARNING, ERROR)
        backend_log_level: 后端 logger 日志级别 (DEBUG, INFO, WARNING, ERROR)，默认 WARNING
    """
    level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    level = level_map.get(log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.setLevel(level)
    
    # 设置后端日志级别环境变量，供 executor 层的 Logger 读取
    os.environ['BACKEND_LOG_LEVEL'] = backend_log_level.upper()
    logger.debug(f"后端日志级别已设置为: {backend_log_level.upper()}")


def _is_shutdown() -> bool:
    """统一的关闭判定：主进程信号标记 或 worker 进程的 shutdown 事件被置位。"""
    return _shutdown_requested or (
        _worker_shutdown_event is not None and _worker_shutdown_event.is_set()
    )


def _init_worker(log_level: str, backend_log_level: str, shutdown_event):
    """ProcessPoolExecutor 的 worker 初始化函数。

    - 在每个 worker 进程中重新配置日志（spawn 子进程不继承父进程的 logging 配置）；
    - 将主进程传入的 multiprocessing.Event 绑定到 worker 全局，供 run_episode 轮询实现快速中断。
    """
    global _worker_shutdown_event
    setup_logging(log_level, backend_log_level)
    _worker_shutdown_event = shutdown_event


def load_agent_config(agent_name: str) -> dict:
    """从 config/agents/ 目录加载指定 Agent 的配置文件

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
    return agent_config


# ==================== AGV 实例解析（复用 auto_train_loop） ====================

def parse_agv_instance(filepath: Path) -> dict:
    """解析 AGV 数据文件

    格式：
    - 第一行: jobs machines agvs points links
    - 接下来 N 行: 作业工序数据
    - 接下来 P 行: 节点信息 (point_id, x, y)
    - 接下来 L 行: 边信息 (link_id, begin, end, weight)
    - 接下来 M 行: 机器绑定 (machine_id, point_id)
    - 最后 A 行: AGV配置 (agv_id, point_id, velocity)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = 0
    first_line = lines[idx].strip().split()
    job_count = int(first_line[0])
    machine_count = int(first_line[1])
    agv_count = int(first_line[2])
    point_count = int(first_line[3])
    link_count = int(first_line[4])
    idx += 1

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

    points = []
    for _ in range(point_count):
        if idx >= len(lines):
            raise ValueError(f"文件过早结束，期望 {point_count} 个节点")
        parts = lines[idx].strip().split()
        if len(parts) != 3:
            raise ValueError(f"节点数据格式错误（第 {idx+1} 行）: {lines[idx].strip()}")
        points.append((int(parts[0]), float(parts[1]), float(parts[2])))
        idx += 1

    links = []
    for _ in range(link_count):
        if idx >= len(lines):
            raise ValueError(f"文件过早结束，期望 {link_count} 条边")
        parts = lines[idx].strip().split()
        if len(parts) != 4:
            raise ValueError(f"边数据格式错误（第 {idx+1} 行）: {lines[idx].strip()}")
        links.append((int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])))
        idx += 1

    machines = []
    for _ in range(machine_count):
        if idx >= len(lines):
            raise ValueError(f"文件过早结束，期望 {machine_count} 台机器")
        parts = lines[idx].strip().split()
        if len(parts) != 2:
            raise ValueError(f"机器绑定数据格式错误（第 {idx+1} 行）: {lines[idx].strip()}")
        machines.append((int(parts[0]), int(parts[1])))
        idx += 1

    agvs = []
    for _ in range(agv_count):
        if idx >= len(lines):
            raise ValueError(f"文件过早结束，期望 {agv_count} 个AGV")
        parts = lines[idx].strip().split()
        if len(parts) != 3:
            raise ValueError(f"AGV配置数据格式错误（第 {idx+1} 行）: {lines[idx].strip()}")
        agvs.append((int(parts[0]), int(parts[1]), float(parts[2])))
        idx += 1

    return {
        "job_count": job_count,
        "machine_count": machine_count,
        "agv_count": agv_count,
        "jobs": jobs,
        "points": points,
        "links": links,
        "machines": machines,
        "agvs": agvs,
    }


# ==================== 配置构建 ====================

def generate_instance_config(parsed_data: dict, uncertain_events: Optional[List[dict]] = None) -> dict:
    """从解析的数据生成 job_config / map_config / event_config

    Args:
        parsed_data: parse_agv_instance() 的输出
        uncertain_events: 可选的预生成不确定性事件时间线
    """
    points = [{"point": {"id": pid, "coordinate": [x, y]}} for pid, x, y in parsed_data["points"]]
    links = [{"link": {"id": lid, "begin": p1, "end": p2}} for lid, p1, p2, _ in parsed_data["links"]]
    machines = [
        {"machine": {"id": mid, "type": "packet_factory.Machine", "point_id": pid}}
        for mid, pid in parsed_data["machines"]
    ]
    agvs = [
        {"agv": {"id": aid, "type": "packet_factory.Agv", "point_id": pid, "velocity": vel, "capacity": 12}}
        for aid, pid, vel in parsed_data["agvs"]
    ]

    all_x = [p[1] for p in parsed_data["points"]]
    all_y = [p[2] for p in parsed_data["points"]]
    width = int(max(all_x) + 5) if all_x else 20
    height = int(max(all_y) + 5) if all_y else 30

    jobs_yaml = []
    for job_id, operations in parsed_data["jobs"]:
        job_entry = {"job": {"id": job_id, "operations": []}}
        for op_idx, machine_options in enumerate(operations):
            op_entry = {"operation": {"id": op_idx, "machines": [{"id": m, "time": d} for m, d in machine_options]}}
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
            "packet_factory.JOB_ADD",
        ]
    }

    # 如果有预生成的不确定性事件，注入 event_timeline
    if uncertain_events:
        event_config["event_timeline"] = [{"event": e} for e in uncertain_events]

    return {
        "event_config": event_config,
        "job_config": {"jobs": jobs_yaml},
        "map_config": {"width": width, "height": height, "points": points, "machines": machines, "links": links, "agvs": agvs},
    }


# ==================== 不确定性事件加载 ====================

def load_uncertain_events_for_instance(instance_path: Path, scenario: str) -> Optional[List[dict]]:
    """
    加载与实例对应的不确定性事件数据

    Args:
        instance_path: AGV 实例文件路径（绝对路径，来自 DATA_DIR 的 glob 结果）
        scenario: 场景名称

    Returns:
        event_timeline 列表，若不存在返回 None
    """
    from dataset import UNCERTAIN_EVENTS_DIR

    instance_path = Path(instance_path)

    # 解析相对路径
    try:
        relative = instance_path.relative_to(DATA_DIR)
    except ValueError:
        relative = instance_path

    event_filename = instance_path.stem + "_events.json"
    event_relative = relative.parent / event_filename

    event_path = Path(UNCERTAIN_EVENTS_DIR) / scenario / event_relative

    if not event_path.exists():
        return None

    with open(event_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("event_timeline", [])


def build_config(agent_key: str, instance_config: dict, base_yaml_path: Optional[str] = None, time_limit: int = 30, num_workers: Optional[int] = None, device: Optional[str] = None) -> dict:
    """构建完整的 bootstrap 配置

    从 config/agents/{agent_key}.yaml 加载 Agent 超参数，
    再用 AGENT_CONFIGS 中的推理模式 identity 字段覆盖。

    Args:
        agent_key: Agent 标识键（DualDRLAgent / ORToolsAgent / ORToolsBatchAgent 等）
        instance_config: generate_instance_config() 的输出
        base_yaml_path: 可选的自定义基础 YAML 路径
        time_limit: OR-Tools 求解时间限制
        num_workers: OR-Tools 每次求解的内部线程数；仅对 OR_TOOLS_AGENTS 生效，
            通过 agent_initializer 的 extra_kwargs 机制透传到 ORToolsOptimizer。
        device: 可选的计算设备覆盖（'auto'/'cpu'/'cuda'），优先于 agent YAML 中的 device 配置；
            None 表示不覆盖，沿用 YAML 中的 device。仅对 DRL agent 有意义（OR-Tools 始终 CPU）。
    """
    yaml_path = base_yaml_path or str(DEFAULT_CONFIG_PATH)
    with open(yaml_path, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    config = copy.deepcopy(template["config"])

    # 从 per-agent 配置文件加载完整 agent 参数（超参数 + identity 字段，含 mode）
    agent_full_config = load_agent_config(agent_key)

    # 填充 agent 段
    config["simulation"]["agent"].update(agent_full_config)

    # Benchmark 覆盖：始终使用 inference + backend 模式
    config["simulation"]["ui_mode"] = "backend"
    config["simulation"]["task_mode"] = "inference"

    # 用 AGENT_CONFIGS 覆盖特定参数
    agent_cfg = AGENT_CONFIGS[agent_key]
    if "time_limit_seconds" in agent_cfg:
        config["simulation"]["agent"]["time_limit_seconds"] = time_limit
    if "model_path" in agent_cfg:
        config["simulation"]["agent"]["model_path"] = agent_cfg["model_path"]

    # 计算设备覆盖：CLI --device 优先于 agent YAML 中的 device 配置
    if device is not None:
        config["simulation"]["agent"]["device"] = device

    # OR-Tools 内部线程数：仅对 OR_TOOLS_AGENTS 注入，避免超额订阅 CPU（并行数×num_workers≈核数）
    if num_workers is not None and agent_key in OR_TOOLS_AGENTS:
        config["simulation"]["agent"]["num_workers"] = num_workers

    # 注入实例数据
    config["simulation"]["job_config"] = instance_config["job_config"]
    config["simulation"]["map_config"] = instance_config["map_config"]
    config["simulation"]["event_config"] = instance_config["event_config"]

    return config


# ==================== Episode 运行 ====================

def run_episode(config: dict, timeout: int = 600, bootstrap_lock: Optional[threading.Lock] = None) -> dict:
    """运行一次完整 episode 并返回结果

    直接调用 executor 层，不走 HTTP。

    Args:
        config: 完整的 bootstrap 配置字典
        timeout: 单次 episode 超时秒数
        bootstrap_lock: 线程模式下用于串行化 bootstrap() 的锁（进程模式传 None）。
            bootstrap() 会读写进程级全局 component_registry['config']（BackendMapLoader
            在 create_context 阶段读取并缓存），多线程并发调用会竞态；env.reset() 使用缓存
            后的实例属性，可放锁外以保留更多并行度。
    """
    from executor.packet_factory.lifecycle.bootstrap import bootstrap
    from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import EnvStatus

    # 线程模式下串行化 bootstrap（保护全局配置）；进程模式无需锁
    if bootstrap_lock is not None:
        with bootstrap_lock:
            env, agent = bootstrap(config)
    else:
        env, agent = bootstrap(config)

    # 重置环境（加载实例数据，初始化 jobs/machines/agvs）
    env.reset()

    # headless 模式
    env.status = EnvStatus.RUNNING
    if env.env_visualizer is not None:
        env.env_visualizer = None

    start_time = time.time()
    step_count = 0

    try:
        while not env.env_is_finished():
            if _is_shutdown():
                logger.info("收到中断信号，正在停止当前 episode...")
                return {"status": "interrupted", "makespan": None, "decision_stats": {}, "steps": step_count, "elapsed": time.time() - start_time}

            if time.time() - start_time > timeout:
                logger.warning(f"Episode 超时 ({timeout}s)，已执行 {step_count} 步")
                return {"status": "timeout", "makespan": None, "decision_stats": {}, "steps": step_count, "elapsed": time.time() - start_time}

            actions = env.action_space(agent)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            step_count += 1

            # DualDRLAgent 在 inference 模式下不需要 train，但保持兼容
            if hasattr(agent, "update") and agent.task_mode == "training":
                agent.update(obs, rewards)

        makespan = env.env_timeline
        decision_stats = agent.get_decision_stats() if hasattr(agent, "get_decision_stats") else {}
        elapsed = time.time() - start_time

        return {"status": "completed", "makespan": makespan, "decision_stats": decision_stats, "steps": step_count, "elapsed": elapsed}

    except Exception as e:
        logger.error(f"Episode 运行异常: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "makespan": None, "decision_stats": {}, "steps": step_count, "elapsed": time.time() - start_time, "error": str(e)}


def run_trial(task: dict) -> dict:
    """并行 worker 的入口函数（ProcessPoolExecutor / ThreadPoolExecutor 的 target）。

    接收一个 task 字典，运行一次 episode，返回可直接写入 JSONL 的 record。
    顶层函数便于 spawn 子进程作为 __main__.run_trial pickle 引用。

    Args:
        task: {config, timeout, agent, family, instance, run, num_runs, bootstrap_lock}
    """
    agent_key = task["agent"]
    family = task["family"]
    instance_name = task["instance"]
    run = task["run"]
    num_runs = task["num_runs"]
    logger.info(f"[worker] {agent_key} | {family}/{instance_name} | run {run}/{num_runs} 开始")

    try:
        result = run_episode(task["config"], timeout=task["timeout"], bootstrap_lock=task.get("bootstrap_lock"))
    except Exception as e:
        # 兜底 bootstrap 期或其它未捕获异常，避免单个 worker 崩溃终止整个调度
        logger.error(f"[worker] {agent_key} | {family}/{instance_name} | run {run} 异常: {e}")
        import traceback
        traceback.print_exc()
        result = {"status": "error", "makespan": None, "decision_stats": {}, "steps": 0, "elapsed": 0.0, "error": str(e)}

    record = {
        "agent": agent_key,
        "family": family,
        "instance": instance_name,
        "run": run,
        "status": result["status"],
        "makespan": result.get("makespan"),
        "decision_stats": result.get("decision_stats", {}),
        "steps": result.get("steps", 0),
        "elapsed": result.get("elapsed", 0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if "error" in result:
        record["error"] = result["error"]
    return record


# ==================== 结果管理 ====================

def load_completed_trials(results_path: Path) -> Set[Tuple[str, str, str, int]]:
    """读取已有结果，返回 (agent, family, instance, run) 的集合。

    status 为 "error" 的记录视为未完成：不计入 completed 集合（重跑时会重新运行），
    并从 results.jsonl 中物理删除这些行，避免错误记录堆积。
    """
    completed: Set[Tuple[str, str, str, int]] = set()
    if not results_path.exists():
        return completed

    kept_lines: List[str] = []
    error_count = 0
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                r = json.loads(raw)
            except json.JSONDecodeError:
                # 无法解析的行原样保留，不影响其余流程
                kept_lines.append(raw)
                continue
            if r.get("status") == "error":
                error_count += 1
                continue
            completed.add((r["agent"], r["family"], r["instance"], r["run"]))
            kept_lines.append(raw)

    # 若存在 error 记录，重写 JSONL（删除 error 行）
    if error_count > 0:
        with open(results_path, "w", encoding="utf-8") as f:
            for raw in kept_lines:
                f.write(raw + "\n")
        logger.info(f"已删除 {error_count} 条 error 记录，将重跑对应 trial")

    return completed


def append_result(results_path: Path, record: dict):
    """追加一条结果到 JSONL 文件"""
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==================== 数据集发现 ====================

def is_small_instance(filepath: Path, max_jobs: int = 15, max_machines: int = 10) -> bool:
    """判断实例是否属于小规模数据集（读取首行即可判断）

    Args:
        filepath: 实例文件路径
        max_jobs: 最大作业数阈值
        max_machines: 最大机器数阈值
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().split()
        job_count = int(first_line[0])
        machine_count = int(first_line[1])
        return job_count <= max_jobs and machine_count <= max_machines
    except Exception:
        return False


def discover_instances(families: List[str], small_only: bool = False, max_jobs: int = 15, max_machines: int = 10) -> Dict[str, List[Path]]:
    """按族发现 AGV 实例文件（递归搜索子目录）

    Args:
        families: 数据集族列表
        small_only: 是否只保留小规模实例
        max_jobs: 小规模实例的最大作业数阈值
        max_machines: 小规模实例的最大机器数阈值
    """
    result = {}
    for family in families:
        family_dir = DATA_DIR / family
        if not family_dir.exists():
            logger.warning(f"数据集族目录不存在: {family_dir}")
            continue
        # 使用 rglob 递归搜索所有子目录中的 _agv.txt 文件
        files = sorted(family_dir.rglob("*_agv.txt"))
        if small_only:
            files = [f for f in files if is_small_instance(f, max_jobs, max_machines)]
        if files:
            result[family] = files
            logger.info(f"  {family}: {len(files)} 个实例" + (" (仅小规模)" if small_only else ""))
        else:
            logger.warning(f"  {family}: 未找到实例文件")
    return result


# ==================== 主实验循环 ====================

def run_benchmark(args):
    """运行完整的 benchmark 实验"""
    global _shutdown_requested

    # 确定实验 ID 和目录
    experiment_id = args.experiment_id or time.strftime("bench_%Y%m%d_%H%M%S")
    exp_dir = RESULTS_ROOT / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_path = exp_dir / "results.jsonl"

    # 保存实验配置快照
    config_snapshot = {
        "experiment_id": experiment_id,
        "agents": args.agents,
        "families": args.families,
        "runs_drl": args.runs_drl,
        "runs_opt": args.runs_opt,
        "timeout": args.timeout,
        "time_limit": args.time_limit,
        "config_yaml": args.config_yaml,
        "small_only": args.small_only,
        "small_max_jobs": args.small_max_jobs,
        "small_max_machines": args.small_max_machines,
        "workers": args.workers,
        "parallel_mode": args.parallel_mode,
        "ortools_workers": args.ortools_workers,
        "device": args.device,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(exp_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    # 加载已完成的 trial
    completed = load_completed_trials(results_path)
    logger.info(f"已完成 {len(completed)} 个 trial，将跳过")

    # 发现实例
    logger.info("发现数据集实例...")
    family_instances = discover_instances(
        args.families,
        small_only=args.small_only,
        max_jobs=args.small_max_jobs,
        max_machines=args.small_max_machines,
    )
    if not family_instances:
        logger.error("未找到任何实例文件")
        return

    # ---- OR-Tools 内部线程数（避免并行时超额订阅 CPU）----
    # 使 并行数(workers) × 每解线程数(num_workers) ≈ CPU 核数
    if args.ortools_workers is not None:
        ortools_nw = args.ortools_workers
    elif args.workers > 1:
        ortools_nw = max(1, (os.cpu_count() or 1) // args.workers)
    else:
        ortools_nw = 4
    logger.info(f"并行配置: workers={args.workers}, mode={args.parallel_mode}, OR-Tools num_workers={ortools_nw}")

    # ---- 预解析 + 缓存每个实例的 instance_config（同一实例的多个 run 复用）----
    instance_config_cache: Dict[str, dict] = {}

    def get_instance_config(instance_file: Path) -> Optional[dict]:
        key = str(instance_file)
        if key in instance_config_cache:
            return instance_config_cache[key]
        try:
            parsed = parse_agv_instance(instance_file)
        except Exception as e:
            logger.error(f"解析失败 {instance_file}: {e}")
            return None
        instance_config = generate_instance_config(parsed)
        if args.uncertain_scenario:
            uncertain_events = load_uncertain_events_for_instance(instance_file, args.uncertain_scenario)
            if uncertain_events:
                instance_config = generate_instance_config(parsed, uncertain_events=uncertain_events)
        instance_config_cache[key] = instance_config
        return instance_config

    # ---- 构建 task 列表（跳过已完成）----
    tasks: List[dict] = []
    skipped = 0
    for agent_key in args.agents:
        num_runs = args.runs_drl if agent_key in DRL_AGENTS else args.runs_opt
        for family, instances in family_instances.items():
            for instance_file in instances:
                instance_name = instance_file.stem.replace("_agv", "")
                instance_config = get_instance_config(instance_file)
                if instance_config is None:
                    continue
                for run in range(1, num_runs + 1):
                    if (agent_key, family, instance_name, run) in completed:
                        skipped += 1
                        continue
                    config = build_config(
                        agent_key, instance_config,
                        base_yaml_path=args.config_yaml,
                        time_limit=args.time_limit,
                        num_workers=(ortools_nw if agent_key in OR_TOOLS_AGENTS else None),
                        device=args.device,
                    )
                    tasks.append({
                        "config": config,
                        "timeout": args.timeout,
                        "agent": agent_key,
                        "family": family,
                        "instance": instance_name,
                        "run": run,
                        "num_runs": num_runs,
                        "bootstrap_lock": None,  # 稍后按并行模式注入
                    })

    total_trials = len(tasks)
    logger.info(f"待运行 trial 数: {total_trials}（跳过 {skipped} 个已完成）")

    if total_trials == 0:
        logger.info("所有 trial 均已完成，无需运行")
        return

    # ---- 创建 pool + shutdown_event ----
    if args.parallel_mode == "process":
        # 显式使用 spawn 上下文，并让 Event 与 pool 共用同一上下文；
        # 否则 Linux 默认 fork 上下文的 Event 传入 spawn pool 会触发
        # "A SemLock created in a fork context is being shared with a process in a spawn context"。
        # spawn 还能避免 fork + CUDA/线程导入导致的死锁。
        ctx = multiprocessing.get_context("spawn")
        shutdown_event = ctx.Event()
        bootstrap_lock = None  # 进程模式各进程独立全局，无需锁；且 threading.Lock 不可 pickle
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.log_level, args.backend_log_level, shutdown_event),
            mp_context=ctx,
        )
    else:  # thread
        shutdown_event = threading.Event()
        bootstrap_lock = threading.Lock()  # 串行化 bootstrap()，保护全局 component_registry['config']
        pool = ThreadPoolExecutor(max_workers=args.workers)
        for t in tasks:
            t["bootstrap_lock"] = bootstrap_lock

    # ---- 注册信号处理器（闭包捕获 shutdown_event，通知 worker 进程快速退出）----
    def signal_handler(sig, frame):
        global _shutdown_requested
        _shutdown_requested = True
        shutdown_event.set()
        logger.info("\n收到中断信号，停止派发新 trial，等待运行中的 episode 退出...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ---- 滚动窗口并行调度 ----
    trial_count = 0
    cap = max(args.workers * 4, args.workers)  # 在途 future 上限，控制内存
    task_iter = iter(tasks)
    pending: set = set()
    fut_to_task: dict = {}
    drain_cancelled = False

    def submit_one() -> bool:
        try:
            t = next(task_iter)
        except StopIteration:
            return False
        fut = pool.submit(run_trial, t)
        pending.add(fut)
        fut_to_task[fut] = t
        return True

    # 预先派发到 cap
    for _ in range(min(cap, len(tasks))):
        submit_one()

    try:
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.discard(fut)
                t = fut_to_task.pop(fut)
                trial_count += 1
                agent_key = t["agent"]
                family = t["family"]
                instance_name = t["instance"]
                run = t["run"]
                num_runs = t["num_runs"]

                try:
                    record = fut.result()
                except Exception as e:
                    logger.error(f"[{trial_count}/{total_trials}] {agent_key} | {family}/{instance_name} | run {run}/{num_runs} 调度异常: {e}")
                    record = {
                        "agent": agent_key, "family": family, "instance": instance_name, "run": run,
                        "status": "error", "makespan": None, "decision_stats": {}, "steps": 0, "elapsed": 0.0,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "error": str(e),
                    }

                append_result(results_path, record)

                # 日志输出
                if record["status"] == "completed" and record.get("makespan") is not None:
                    makespan = record["makespan"]
                    avg_dt = record.get("decision_stats", {}).get("average_decision_time", 0)
                    logger.info(f"[{trial_count}/{total_trials}] {agent_key} | {family}/{instance_name} | run {run}/{num_runs} -> makespan={makespan:.2f}, avg_decision_time={avg_dt:.4f}s, steps={record['steps']}, elapsed={record['elapsed']:.2f}s")
                else:
                    logger.warning(f"[{trial_count}/{total_trials}] {agent_key} | {family}/{instance_name} | run {run}/{num_runs} -> {record['status']}")

                # 未中断则补充派发一个，维持在途数量
                if not _shutdown_requested:
                    submit_one()

            # 中断后：通知 worker 退出、取消尚未开始的 future 并记录 cancelled，继续 drain 运行中的
            if _shutdown_requested and not drain_cancelled:
                drain_cancelled = True
                shutdown_event.set()
                for fut in list(pending):
                    if fut.cancel():
                        pending.discard(fut)
                        ct = fut_to_task.pop(fut, None)
                        if ct is not None:
                            trial_count += 1
                            append_result(results_path, {
                                "agent": ct["agent"], "family": ct["family"], "instance": ct["instance"], "run": ct["run"],
                                "status": "cancelled", "makespan": None, "decision_stats": {}, "steps": 0, "elapsed": 0.0,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            })
                            logger.warning(f"[{trial_count}/{total_trials}] {ct['agent']} | {ct['family']}/{ct['instance']} | run {ct['run']}/{ct['num_runs']} -> cancelled")
    except KeyboardInterrupt:
        # 二次中断兜底：取消未开始的任务并直接关闭（运行中的结果可能丢失）
        _shutdown_requested = True
        shutdown_event.set()
        logger.info("KeyboardInterrupt: 取消未开始的 trial 并退出...")
        for fut in list(pending):
            fut.cancel()
    finally:
        pool.shutdown(cancel_futures=True, wait=True)


    # 总结
    logger.info(f"\n{'='*60}")
    logger.info(f"实验完成: {experiment_id}")
    logger.info(f"新运行: {trial_count}, 跳过: {skipped}")
    if _shutdown_requested:
        logger.info("（被中断）再次运行相同命令可继续")
    logger.info(f"结果保存在: {results_path}")
    logger.info(f"{'='*60}")


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="自动化调度算法对比实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行全部 agent × 全部数据集
  uv run python scripts/run_benchmark.py

  # 只测试 ORToolsAgent 在 kacem 数据集上
  uv run python scripts/run_benchmark.py --agents ORToolsAgent --families kacem --runs-opt 1

  # 使用自定义 YAML 模板
  uv run python scripts/run_benchmark.py --config-yaml my_config.yaml

  # 指定实验 ID（用于续跑）
  uv run python scripts/run_benchmark.py --experiment-id bench_20260519

  # 只运行小规模数据集（默认 jobs<=15, machines<=10），方便与全局最优解对比
  uv run python scripts/run_benchmark.py --small-only

  # 自定义小规模阈值
  uv run python scripts/run_benchmark.py --small-only --small-max-jobs 10 --small-max-machines 8

  # 多进程并行加速（默认使用全部 CPU 核）
  uv run python scripts/run_benchmark.py --agents ORToolsAgent --families kacem -j 8

  # 多线程模式（轻量，OR-Tools 求解释放 GIL 时收益明显）
  uv run python scripts/run_benchmark.py --parallel-mode thread -j 8

  # 显式控制 OR-Tools 每 solver 线程数，避免超额订阅
  uv run python scripts/run_benchmark.py -j 8 --ortools-workers 2

  # 强制 DRL agent 使用 CPU（覆盖 YAML 中的 device: cuda，免改配置文件）
  uv run python scripts/run_benchmark.py --agents DualDRLAgent --device cpu
        """,
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(AGENT_CONFIGS.keys()),
        choices=list(AGENT_CONFIGS.keys()),
        help="要测试的 agent 列表 (默认: 全部)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=ALL_FAMILIES,
        choices=ALL_FAMILIES,
        help="数据集族列表 (默认: 全部)",
    )
    parser.add_argument("--runs-drl", type=int, default=5, help="DualDRLAgent 每实例评估次数 (默认: 5)")
    parser.add_argument("--runs-opt", type=int, default=3, help="OR-Tools agent 每实例运行次数 (默认: 3)")
    parser.add_argument("--config-yaml", type=str, default=None, help="可选的自定义基础 YAML 配置路径")
    parser.add_argument("--experiment-id", type=str, default=None, help="实验标识（默认自动生成时间戳）")
    parser.add_argument("--timeout", type=int, default=600, help="单次 episode 超时秒数 (默认: 600)")
    parser.add_argument("--time-limit", type=int, default=30, help="OR-Tools 求解时间限制秒数 (默认: 30)")
    parser.add_argument("-j", "--workers", type=int, default=8,
                        help="并行 worker 数 (默认: 8)。process 模式下每个 trial 在独立进程运行；thread 模式下在独立线程运行")
    parser.add_argument("--parallel-mode", type=str, choices=["process", "thread"], default="process",
                        help="并行模式 (默认: process)。process=多进程(真多核,绕过GIL); thread=多线程(轻量,但受GIL限制,且 bootstrap 会被锁串行化)")
    parser.add_argument("--ortools-workers", type=int, default=None,
                        help="OR-Tools 每个 solver 的内部线程数 (默认: 自动=cpu_count//workers, workers=1 时为 4)。显式指定时可配合 --workers 精确控制总线程数")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="前端脚本日志级别")
    parser.add_argument("--backend-log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="后端 logger 日志级别 (默认: WARNING)")
    parser.add_argument("--small-only", action="store_true",
                        help="只运行小规模数据集，方便与全局最优解对比")
    parser.add_argument("--small-max-jobs", type=int, default=15,
                        help="小规模实例的最大作业数阈值 (默认: 15)")
    parser.add_argument("--small-max-machines", type=int, default=10,
                        help="小规模实例的最大机器数阈值 (默认: 10)")
    parser.add_argument("--uncertain-scenario", type=str, default=None,
                        help="不确定性事件场景名（如 default, heavy），加载 dataset/uncertain-events/<scenario>/ 下的预生成事件数据")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"],
                        help="计算设备覆盖：强制指定 DRL agent 的 device，优先于 agent YAML 中的 device 配置 "
                             "(默认: None=使用 YAML；cpu=强制 CPU；cuda=强制 GPU 不可用时回退 CPU；auto=自动选择)。"
                             "OR-Tools agent 始终 CPU，此参数对其无影响")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level, args.backend_log_level)

    logger.info("=" * 60)
    logger.info("SkyEngine Benchmark Experiment")
    logger.info(f"Agents: {args.agents}")
    logger.info(f"Families: {args.families}")
    logger.info(f"Runs (DRL): {args.runs_drl}, Runs (Opt): {args.runs_opt}")
    logger.info(f"Timeout: {args.timeout}s, Time Limit: {args.time_limit}s")
    logger.info(f"Backend Log Level: {args.backend_log_level.upper()}")
    logger.info(f"Workers: {args.workers}, Parallel Mode: {args.parallel_mode}, OR-Tools workers: {args.ortools_workers if args.ortools_workers is not None else 'auto(cpu_count//workers)'}")
    if args.small_only:
        logger.info(f"Small-only mode: max_jobs={args.small_max_jobs}, max_machines={args.small_max_machines}")
    if args.uncertain_scenario:
        logger.info(f"Uncertain scenario: {args.uncertain_scenario}")
    if args.config_yaml:
        logger.info(f"Base YAML: {args.config_yaml}")
    if args.device:
        logger.info(f"Device override: {args.device} (覆盖 DRL agent YAML 中的 device 配置)")
    logger.info("=" * 60)

    run_benchmark(args)
