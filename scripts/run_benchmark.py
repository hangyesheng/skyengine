"""
自动化调度算法对比实验脚本

功能：
1. 直接调用 executor 层运行 DualDRLAgent / ORToolsAgent / ORToolsBatchAgent
2. 在 agv-instances 各数据集族上运行实验
3. 实时保存结果到 JSONL（支持中断续跑）
4. 自动构建配置（支持 --config-yaml 覆盖基础模板）

用法：
    uv run python scripts/run_benchmark.py --agents ORToolsAgent --families kacem --runs-opt 1
    uv run python scripts/run_benchmark.py --agents DualDRLAgent ORToolsAgent ORToolsBatchAgent
    uv run python scripts/run_benchmark.py --config-yaml custom.yaml --families brandimarte
    uv run python scripts/run_benchmark.py --log-level DEBUG --backend-log-level INFO

中断后续跑：
    再次运行相同命令即可，已完成的 trial 会自动跳过

日志级别控制：
    --log-level: 控制前端脚本的日志输出级别 (默认: INFO)
    --backend-log-level: 控制后端 logger 的日志输出级别 (默认: WARNING)
"""

import argparse
import copy
import json
import logging
import os
import signal
import sys
import time
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

# === 日志 ===
logger = logging.getLogger("benchmark")

# === Agent 配置映射 ===
AGENT_CONFIGS = {
    "DualDRLAgent": {
        "agent_name": "packet_factory.DualDRLAgent",
        "mode": "drl",
        "task_mode": "inference",
        "model_path": "./training_logs/models/DualDRLAgent/agent_model.json",
    },
    "ORToolsAgent": {
        "agent_name": "packet_factory.ORToolsAgent",
        "mode": "drl",
        "task_mode": "inference",
        "time_limit_seconds": 30,
    },
    "ORToolsBatchAgent": {
        "agent_name": "packet_factory.ORToolsBatchAgent",
        "mode": "optimization",
        "task_mode": "inference",
        "time_limit_seconds": 60,
    },
}

ALL_FAMILIES = ["barnes", "behnke", "brandimarte", "dauzere", "fattahi", "hurink", "kacem"]

# === 全局中断标记 ===
_shutdown_requested = False


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

def generate_instance_config(parsed_data: dict) -> dict:
    """从解析的数据生成 job_config / map_config / event_config"""
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

    return {
        "event_config": {
            "event_type": [
                "packet_factory.JUST_TEST",
                "packet_factory.ENV_PAUSED",
                "packet_factory.ENV_RECOVER",
                "packet_factory.ENV_RESTART",
                "packet_factory.AGV_FAIL",
                "packet_factory.MACHINE_FAIL",
                "packet_factory.JOB_ADD",
            ]
        },
        "job_config": {"jobs": jobs_yaml},
        "map_config": {"width": width, "height": height, "points": points, "machines": machines, "links": links, "agvs": agvs},
    }


def build_config(agent_key: str, instance_config: dict, base_yaml_path: Optional[str] = None, time_limit: int = 30) -> dict:
    """构建完整的 bootstrap 配置

    Args:
        agent_key: Agent 标识键（DualDRLAgent / ORToolsAgent / ORToolsBatchAgent）
        instance_config: generate_instance_config() 的输出
        base_yaml_path: 可选的自定义基础 YAML 路径
        time_limit: OR-Tools 求解时间限制
    """
    yaml_path = base_yaml_path or str(DEFAULT_CONFIG_PATH)
    with open(yaml_path, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    config = copy.deepcopy(template["config"])
    agent_cfg = AGENT_CONFIGS[agent_key]

    # 覆盖 agent 配置
    config["simulation"]["mode"] = agent_cfg["mode"]
    config["simulation"]["agent"]["agent_name"] = agent_cfg["agent_name"]
    config["simulation"]["agent"]["name"] = agent_key
    config["simulation"]["agent"]["id"] = 1
    config["simulation"]["agent"]["ui_mode"] = "backend"
    config["simulation"]["agent"]["task_mode"] = agent_cfg["task_mode"]

    if "time_limit_seconds" in agent_cfg:
        config["simulation"]["agent"]["time_limit_seconds"] = time_limit
    if "model_path" in agent_cfg:
        config["simulation"]["agent"]["model_path"] = agent_cfg["model_path"]
    config["simulation"]["agent"]["fallback_enabled"] = True

    # 注入实例数据
    config["simulation"]["job_config"] = instance_config["job_config"]
    config["simulation"]["map_config"] = instance_config["map_config"]
    config["simulation"]["event_config"] = instance_config["event_config"]

    return config


# ==================== Episode 运行 ====================

def run_episode(config: dict, timeout: int = 600) -> dict:
    """运行一次完整 episode 并返回结果

    直接调用 executor 层，不走 HTTP。
    """
    from executor.packet_factory.lifecycle.bootstrap import bootstrap
    from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import EnvStatus

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
            if _shutdown_requested:
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


# ==================== 结果管理 ====================

def load_completed_trials(results_path: Path) -> Set[Tuple[str, str, str, int]]:
    """读取已有结果，返回 (agent, family, instance, run) 的集合"""
    completed = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    completed.add((r["agent"], r["family"], r["instance"], r["run"]))
                except json.JSONDecodeError:
                    continue
    return completed


def append_result(results_path: Path, record: dict):
    """追加一条结果到 JSONL 文件"""
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==================== 数据集发现 ====================

def discover_instances(families: List[str]) -> Dict[str, List[Path]]:
    """按族发现 AGV 实例文件（递归搜索子目录）"""
    result = {}
    for family in families:
        family_dir = DATA_DIR / family
        if not family_dir.exists():
            logger.warning(f"数据集族目录不存在: {family_dir}")
            continue
        # 使用 rglob 递归搜索所有子目录中的 _agv.txt 文件
        files = sorted(family_dir.rglob("*_agv.txt"))
        if files:
            result[family] = files
            logger.info(f"  {family}: {len(files)} 个实例")
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
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(exp_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    # 加载已完成的 trial
    completed = load_completed_trials(results_path)
    logger.info(f"已完成 {len(completed)} 个 trial，将跳过")

    # 发现实例
    logger.info("发现数据集实例...")
    family_instances = discover_instances(args.families)
    if not family_instances:
        logger.error("未找到任何实例文件")
        return

    # 统计总 trial 数
    total_trials = 0
    for agent_key in args.agents:
        num_runs = args.runs_drl if agent_key == "DualDRLAgent" else args.runs_opt
        for family, instances in family_instances.items():
            for instance_file in instances:
                instance_name = instance_file.stem.replace("_agv", "")
                for run in range(1, num_runs + 1):
                    if (agent_key, family, instance_name, run) not in completed:
                        total_trials += 1

    logger.info(f"待运行 trial 数: {total_trials}")

    if total_trials == 0:
        logger.info("所有 trial 均已完成，无需运行")
        return

    # 注册信号处理器
    def signal_handler(sig, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info("\n收到中断信号，等待当前 episode 完成后退出...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 运行实验
    trial_count = 0
    skipped = 0

    for agent_key in args.agents:
        if _shutdown_requested:
            break

        num_runs = args.runs_drl if agent_key == "DualDRLAgent" else args.runs_opt
        agent_cfg = AGENT_CONFIGS[agent_key]

        for family, instances in family_instances.items():
            if _shutdown_requested:
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"Agent: {agent_key} | Family: {family} | Runs: {num_runs}")
            logger.info(f"{'='*60}")

            for instance_file in instances:
                if _shutdown_requested:
                    break

                instance_name = instance_file.stem.replace("_agv", "")

                # 解析实例
                try:
                    parsed = parse_agv_instance(instance_file)
                except Exception as e:
                    logger.error(f"解析失败 {instance_file}: {e}")
                    continue

                instance_config = generate_instance_config(parsed)

                for run in range(1, num_runs + 1):
                    if _shutdown_requested:
                        break

                    # 检查是否已完成
                    if (agent_key, family, instance_name, run) in completed:
                        skipped += 1
                        continue

                    # 构建配置
                    config = build_config(agent_key, instance_config, base_yaml_path=args.config_yaml, time_limit=args.time_limit)

                    # 运行 episode
                    trial_count += 1
                    logger.info(f"[{trial_count}/{total_trials}] {agent_key} | {family}/{instance_name} | run {run}/{num_runs}")

                    result = run_episode(config, timeout=args.timeout)

                    # 记录结果
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

                    append_result(results_path, record)

                    # 日志输出
                    if result["status"] == "completed":
                        makespan = result["makespan"]
                        avg_dt = result.get("decision_stats", {}).get("average_decision_time", 0)
                        logger.info(f"  -> makespan={makespan:.2f}, avg_decision_time={avg_dt:.4f}s, steps={result['steps']}, elapsed={result['elapsed']:.2f}s")
                    else:
                        logger.warning(f"  -> {result['status']}")

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
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="前端脚本日志级别")
    parser.add_argument("--backend-log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="后端 logger 日志级别 (默认: WARNING)")

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
    if args.config_yaml:
        logger.info(f"Base YAML: {args.config_yaml}")
    logger.info("=" * 60)

    run_benchmark(args)
