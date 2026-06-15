"""
不确定性事件数据集生成脚本

功能：
1. 读取 dataset/agv-instances 下的所有实例文件
2. 根据实例特征（机器数、AGV数、预估makespan）随机生成不确定性事件时间线
3. 输出 JSON 格式事件文件到 dataset/uncertain-events/<scenario>/ 目录
4. 支持多种预设场景和自定义参数
5. 使用固定随机种子确保可复现性

支持的事件类型：
- MACHINE_FAIL / MACHINE_RECOVER: 机器故障与恢复（配对）
- AGV_FAIL / AGV_RECOVER: AGV故障与恢复（配对）
- JOB_ADD: 新增作业

用法：
    python scripts/generate_uncertainty_events.py
    python scripts/generate_uncertainty_events.py --seed 42
    python scripts/generate_uncertainty_events.py --scenario heavy
    python scripts/generate_uncertainty_events.py --families kacem brandimarte
    python scripts/generate_uncertainty_events.py --machine-fail-prob 0.3
    python scripts/generate_uncertainty_events.py --scenario default --families kacem --seed 42
"""

import argparse
import hashlib
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# === 路径常量 ===
DATA_DIR = PROJECT_ROOT / "dataset" / "agv-instances"
UNCERTAIN_EVENTS_DIR = PROJECT_ROOT / "dataset" / "uncertain-events"

logger = logging.getLogger("generate_uncertainty_events")

# ==================== 预设场景 ====================

SCENARIOS = {
    "light": {
        "machine_fail_prob": 0.05,
        "agv_fail_prob": 0.03,
        "job_add_prob": 0.00,
        "machine_fail_duration_range": [3, 8],
        "agv_fail_duration_range": [2, 6],
        "timeline_span_ratio": 0.8,
        "min_events": 1,
        "description": "轻微扰动：少量机器/AGV故障",
    },
    "default": {
        "machine_fail_prob": 0.15,
        "agv_fail_prob": 0.10,
        "job_add_prob": 0.05,
        "machine_fail_duration_range": [3, 10],
        "agv_fail_duration_range": [2, 8],
        "timeline_span_ratio": 0.8,
        "min_events": 2,
        "description": "中等扰动：机器和AGV故障 + 少量新作业",
    },
    "heavy": {
        "machine_fail_prob": 0.30,
        "agv_fail_prob": 0.20,
        "job_add_prob": 0.10,
        "machine_fail_duration_range": [5, 15],
        "agv_fail_duration_range": [3, 12],
        "timeline_span_ratio": 0.85,
        "min_events": 3,
        "description": "强扰动：频繁故障 + 较多新作业",
    },
    "machine_only": {
        "machine_fail_prob": 0.25,
        "agv_fail_prob": 0.00,
        "job_add_prob": 0.00,
        "machine_fail_duration_range": [3, 12],
        "agv_fail_duration_range": [2, 8],
        "timeline_span_ratio": 0.8,
        "min_events": 2,
        "description": "仅机器故障场景",
    },
    "agv_only": {
        "machine_fail_prob": 0.00,
        "agv_fail_prob": 0.25,
        "job_add_prob": 0.00,
        "machine_fail_duration_range": [3, 10],
        "agv_fail_duration_range": [2, 10],
        "timeline_span_ratio": 0.8,
        "min_events": 2,
        "description": "仅AGV故障场景",
    },
}

ALL_FAMILIES = ["barnes", "behnke", "brandimarte", "dauzere", "fattahi", "hurink", "kacem"]


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


# ==================== Makespan 下界估算 ====================

def compute_makespan_lower_bound(parsed_data: dict) -> float:
    """
    计算估计的 makespan 下界

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


# ==================== 事件生成 ====================

def generate_machine_fail_events(
    rng: np.random.Generator,
    machine_count: int,
    estimated_makespan: float,
    config: dict,
) -> List[dict]:
    """
    生成机器故障事件（trigger + recover 配对）

    Args:
        rng: 随机数生成器
        machine_count: 机器数量
        estimated_makespan: 预估 makespan
        config: 生成配置

    Returns:
        事件列表
    """
    if machine_count == 0 or config["machine_fail_prob"] == 0:
        return []

    events = []
    timeline_start = estimated_makespan * 0.1
    timeline_end = estimated_makespan * config["timeline_span_ratio"]

    # 故障次数服从泊松分布：期望 = prob * machine_count
    expected_fails = config["machine_fail_prob"] * machine_count
    num_fails = rng.poisson(expected_fails)
    num_fails = max(0, min(num_fails, machine_count * 2))  # 上限保护

    # 随机选择故障机器（允许同一机器多次故障）
    fail_machine_ids = rng.integers(0, machine_count, size=num_fails)

    duration_range = config["machine_fail_duration_range"]

    for mid in fail_machine_ids:
        mid = int(mid)
        # trigger 时间在 [timeline_start, timeline_end * 0.7] 范围内
        # 留出足够时间给 recover 事件
        trigger_t = round(rng.uniform(timeline_start, timeline_end * 0.75), 2)
        # 故障持续时间
        duration = round(rng.uniform(duration_range[0], duration_range[1]), 2)
        recover_t = round(trigger_t + duration, 2)

        # 确保 recover 不超过时间线范围
        if recover_t > timeline_end:
            recover_t = round(timeline_end, 2)

        events.append({
            "timestamp": trigger_t,
            "type": "packet_factory.MACHINE_FAIL",
            "args": ["trigger", {"id": mid}]
        })
        events.append({
            "timestamp": recover_t,
            "type": "packet_factory.MACHINE_FAIL",
            "args": ["recover", {"id": mid}]
        })

    return events


def generate_agv_fail_events(
    rng: np.random.Generator,
    agv_count: int,
    agv_ids: List[int],
    estimated_makespan: float,
    config: dict,
) -> List[dict]:
    """
    生成 AGV 故障事件（trigger + recover 配对）

    Args:
        rng: 随机数生成器
        agv_count: AGV 数量
        agv_ids: AGV ID 列表
        estimated_makespan: 预估 makespan
        config: 生成配置

    Returns:
        事件列表
    """
    if agv_count == 0 or config["agv_fail_prob"] == 0:
        return []

    events = []
    timeline_start = estimated_makespan * 0.1
    timeline_end = estimated_makespan * config["timeline_span_ratio"]

    # 故障次数服从泊松分布
    expected_fails = config["agv_fail_prob"] * agv_count
    num_fails = rng.poisson(expected_fails)
    num_fails = max(0, min(num_fails, agv_count * 2))

    # 随机选择故障 AGV
    fail_agv_indices = rng.integers(0, agv_count, size=num_fails)

    duration_range = config["agv_fail_duration_range"]

    for aidx in fail_agv_indices:
        aidx = int(aidx)
        aid = agv_ids[aidx] if aidx < len(agv_ids) else agv_ids[-1]

        trigger_t = round(rng.uniform(timeline_start, timeline_end * 0.75), 2)
        duration = round(rng.uniform(duration_range[0], duration_range[1]), 2)
        recover_t = round(trigger_t + duration, 2)

        if recover_t > timeline_end:
            recover_t = round(timeline_end, 2)

        events.append({
            "timestamp": trigger_t,
            "type": "packet_factory.AGV_FAIL",
            "args": ["trigger", {"id": aid}]
        })
        events.append({
            "timestamp": recover_t,
            "type": "packet_factory.AGV_FAIL",
            "args": ["recover", {"id": aid}]
        })

    return events


def generate_job_add_events(
    rng: np.random.Generator,
    parsed_data: dict,
    estimated_makespan: float,
    config: dict,
) -> List[dict]:
    """
    生成新增作业事件

    新作业从现有作业中变异生成：随机选择一个已有作业，
    对其工序的加工时间施加随机扰动，分配新的作业 ID。

    Args:
        rng: 随机数生成器
        parsed_data: 解析后的实例数据
        estimated_makespan: 预估 makespan
        config: 生成配置

    Returns:
        事件列表
    """
    if config["job_add_prob"] == 0 or not parsed_data["jobs"]:
        return []

    events = []
    timeline_start = estimated_makespan * 0.2
    timeline_end = estimated_makespan * config["timeline_span_ratio"] * 0.6

    # 新增作业数量
    expected_adds = config["job_add_prob"] * parsed_data["job_count"]
    num_adds = rng.poisson(expected_adds)
    num_adds = max(0, min(num_adds, max(3, parsed_data["job_count"] // 2)))

    # 已有作业 ID 集合，用于分配新 ID
    existing_job_ids = {j[0] for j in parsed_data["jobs"]}
    next_job_id = max(existing_job_ids) + 1 if existing_job_ids else 0

    for i in range(num_adds):
        # 随机选择一个已有作业作为模板
        template_idx = rng.integers(0, len(parsed_data["jobs"]))
        _, template_ops = parsed_data["jobs"][template_idx]

        # 变异：对每个工序的加工时间施加 ±20% 扰动
        new_ops = []
        for op_machines in template_ops:
            new_op_machines = []
            for mid, duration in op_machines:
                # 随机扰动加工时间（±20%），取整
                ratio = 1.0 + rng.uniform(-0.2, 0.2)
                new_duration = max(1, round(duration * ratio))
                new_op_machines.append({"id": mid, "time": new_duration})
            new_ops.append({"id": len(new_ops), "machines": new_op_machines})

        new_job = {
            "id": next_job_id + i,
            "operations": new_ops,
        }

        # 事件时间：在时间线前中期插入，让调度器有足够时间处理
        trigger_t = round(rng.uniform(timeline_start, timeline_end), 2)

        events.append({
            "timestamp": trigger_t,
            "type": "packet_factory.JOB_ADD",
            "args": ["trigger", {"job": new_job}]
        })

    return events


def generate_events_for_instance(
    parsed_data: dict,
    config: dict,
    seed: int,
) -> dict:
    """
    为单个实例生成不确定性事件时间线

    Args:
        parsed_data: 解析后的实例数据
        config: 场景配置（概率、时长范围等）
        seed: 随机种子

    Returns:
        dict: 包含 event_timeline 和元信息的结果字典
    """
    rng = np.random.default_rng(seed)

    machine_count = parsed_data["machine_count"]
    agv_count = parsed_data["agv_count"]
    agv_ids = [a[0] for a in parsed_data["agvs"]]

    estimated_makespan = compute_makespan_lower_bound(parsed_data)

    # 生成各类事件
    machine_events = generate_machine_fail_events(
        rng, machine_count, estimated_makespan, config
    )
    agv_events = generate_agv_fail_events(
        rng, agv_count, agv_ids, estimated_makespan, config
    )
    job_events = generate_job_add_events(
        rng, parsed_data, estimated_makespan, config
    )

    # 合并并按时间排序
    all_events = machine_events + agv_events + job_events
    all_events.sort(key=lambda e: (e["timestamp"], 0 if e["args"][0] == "trigger" else 1))

    # 如果事件数不足 min_events 且场景概率 > 0，强制补充故障事件
    min_events = config.get("min_events", 2)
    if len(all_events) < min_events and min_events > 0:
        timeline_start = estimated_makespan * 0.15
        timeline_end = estimated_makespan * config["timeline_span_ratio"]
        if timeline_end <= timeline_start:
            timeline_end = timeline_start + 5.0

        while len(all_events) < min_events:
            # 优先生成机器故障（通常机器数 >= AGV 数）
            if machine_count > 0 and config["machine_fail_prob"] > 0:
                mid = int(rng.integers(0, machine_count))
                trigger_t = round(rng.uniform(timeline_start, timeline_end * 0.7), 2)
                duration = round(rng.uniform(
                    config["machine_fail_duration_range"][0],
                    config["machine_fail_duration_range"][1],
                ), 2)
                recover_t = round(min(trigger_t + duration, timeline_end), 2)
                all_events.append({
                    "timestamp": trigger_t,
                    "type": "packet_factory.MACHINE_FAIL",
                    "args": ["trigger", {"id": mid}]
                })
                all_events.append({
                    "timestamp": recover_t,
                    "type": "packet_factory.MACHINE_FAIL",
                    "args": ["recover", {"id": mid}]
                })
            elif agv_count > 0 and config["agv_fail_prob"] > 0:
                aid = agv_ids[int(rng.integers(0, agv_count))]
                trigger_t = round(rng.uniform(timeline_start, timeline_end * 0.7), 2)
                duration = round(rng.uniform(
                    config["agv_fail_duration_range"][0],
                    config["agv_fail_duration_range"][1],
                ), 2)
                recover_t = round(min(trigger_t + duration, timeline_end), 2)
                all_events.append({
                    "timestamp": trigger_t,
                    "type": "packet_factory.AGV_FAIL",
                    "args": ["trigger", {"id": aid}]
                })
                all_events.append({
                    "timestamp": recover_t,
                    "type": "packet_factory.AGV_FAIL",
                    "args": ["recover", {"id": aid}]
                })
            else:
                break  # 无可生成的事件类型

        all_events.sort(key=lambda e: (e["timestamp"], 0 if e["args"][0] == "trigger" else 1))

    # 统计事件
    event_counts = {
        "machine_fail_trigger": sum(1 for e in all_events
                                    if e["type"] == "packet_factory.MACHINE_FAIL"
                                    and e["args"][0] == "trigger"),
        "machine_fail_recover": sum(1 for e in all_events
                                    if e["type"] == "packet_factory.MACHINE_FAIL"
                                    and e["args"][0] == "recover"),
        "agv_fail_trigger": sum(1 for e in all_events
                                if e["type"] == "packet_factory.AGV_FAIL"
                                and e["args"][0] == "trigger"),
        "agv_fail_recover": sum(1 for e in all_events
                                if e["type"] == "packet_factory.AGV_FAIL"
                                and e["args"][0] == "recover"),
        "job_add": sum(1 for e in all_events
                       if e["type"] == "packet_factory.JOB_ADD"),
        "total": len(all_events),
    }

    return {
        "instance_info": {
            "job_count": parsed_data["job_count"],
            "machine_count": machine_count,
            "agv_count": agv_count,
            "estimated_makespan": estimated_makespan,
        },
        "generation_config": {
            "seed": seed,
            **config,
        },
        "event_counts": event_counts,
        "event_timeline": all_events,
    }


# ==================== 事件文件 I/O ====================

def save_events_file(output_path: Path, source_instance: str, events_data: dict):
    """保存事件数据到 JSON 文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "source_instance": source_instance,
        **events_data,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_uncertain_events(instance_path: Path, scenario: str = "default") -> Optional[List[dict]]:
    """
    加载与实例对应的不确定性事件数据

    根据实例文件的相对路径，在 uncertain-events/<scenario>/ 下查找对应的事件文件。

    Args:
        instance_path: AGV 实例文件路径（相对于 agv-instances/ 或绝对路径均可）
        scenario: 场景名称

    Returns:
        event_timeline 列表，若不存在返回 None
    """
    # 解析相对路径：支持绝对路径和相对路径
    instance_path = Path(instance_path)
    try:
        relative = instance_path.relative_to(DATA_DIR)
    except ValueError:
        # 可能是相对路径，尝试直接使用
        relative = instance_path

    # 将 _agv.txt 后缀替换为 _agv_events.json
    event_filename = instance_path.stem + "_events.json"
    event_relative = relative.parent / event_filename

    event_path = UNCERTAIN_EVENTS_DIR / scenario / event_relative

    if not event_path.exists():
        return None

    with open(event_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("event_timeline", [])


# ==================== 批量生成 ====================

def generate_all(
    scenario: str,
    config: dict,
    seed: int = 42,
    families: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
):
    """
    批量生成全部实例的不确定性事件数据

    Args:
        scenario: 场景名称
        config: 场景配置
        seed: 基础随机种子
        families: 指定数据集族（None 表示全部）
        output_dir: 输出目录（默认 dataset/uncertain-events/<scenario>/）
    """
    if output_dir is None:
        output_dir = UNCERTAIN_EVENTS_DIR / scenario

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存场景配置
    config_path = output_dir / "generation_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "scenario": scenario,
            "base_seed": seed,
            "config": config,
            "generated_at": str(Path.cwd()),
        }, f, indent=2, ensure_ascii=False)

    # 发现实例文件
    target_families = families or ALL_FAMILIES
    instance_files: List[Path] = []

    for family in target_families:
        family_dir = DATA_DIR / family
        if not family_dir.exists():
            logger.warning(f"数据集族目录不存在: {family_dir}")
            continue
        files = sorted(family_dir.rglob("*_agv.txt"))
        instance_files.extend(files)
        logger.info(f"  {family}: {len(files)} 个实例")

    if not instance_files:
        logger.error("未找到任何实例文件")
        return

    logger.info(f"共发现 {len(instance_files)} 个实例文件")
    logger.info(f"场景: {scenario}, 基础种子: {seed}")
    logger.info(f"输出目录: {output_dir}")

    # 为每个实例生成事件
    success_count = 0
    fail_count = 0

    for instance_path in instance_files:
        try:
            relative = instance_path.relative_to(DATA_DIR)
            event_filename = instance_path.stem + "_events.json"
            output_path = output_dir / relative.parent / event_filename

            # 每个实例使用不同的种子（基于基础种子 + 文件路径的确定性哈希）
            # 注意：使用 hashlib 而非 Python 内置 hash()，因为后者在不同运行间不确定
            path_hash = int(hashlib.md5(str(relative).encode()).hexdigest()[:8], 16)
            instance_seed = seed + (path_hash % 10000)

            parsed_data = parse_agv_instance(instance_path)
            events_data = generate_events_for_instance(parsed_data, config, instance_seed)

            save_events_file(output_path, str(relative), events_data)

            counts = events_data["event_counts"]
            logger.debug(f"  ✓ {relative}: {counts['total']} 个事件 "
                         f"(M_fail={counts['machine_fail_trigger']}, "
                         f"A_fail={counts['agv_fail_trigger']}, "
                         f"J_add={counts['job_add']})")
            success_count += 1

        except Exception as e:
            logger.error(f"  ✗ 处理失败 {instance_path.name}: {e}")
            fail_count += 1

    logger.info(f"\n生成完成: {success_count} 成功, {fail_count} 失败")
    logger.info(f"输出目录: {output_dir}")


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="不确定性事件数据集生成脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/generate_uncertainty_events.py
  python scripts/generate_uncertainty_events.py --seed 42
  python scripts/generate_uncertainty_events.py --scenario heavy
  python scripts/generate_uncertainty_events.py --families kacem brandimarte
  python scripts/generate_uncertainty_events.py --machine-fail-prob 0.3
  python scripts/generate_uncertainty_events.py --scenario default --families kacem --seed 42

预设场景:
  light         轻微扰动 (machine_fail=0.05, agv_fail=0.03)
  default       中等扰动 (machine_fail=0.15, agv_fail=0.10, job_add=0.05)
  heavy         强扰动   (machine_fail=0.30, agv_fail=0.20, job_add=0.10)
  machine_only  仅机器故障 (machine_fail=0.25)
  agv_only      仅AGV故障 (agv_fail=0.25)
        """,
    )

    parser.add_argument(
        '--scenario', type=str, default='default',
        choices=list(SCENARIOS.keys()),
        help='预设场景名称 (默认: default)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='基础随机种子 (默认: 42)'
    )
    parser.add_argument(
        '--families', nargs='+', default=None,
        choices=ALL_FAMILIES,
        help='指定数据集族 (默认: 全部)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='输出目录 (默认: dataset/uncertain-events/<scenario>/)'
    )

    # 自定义概率参数（覆盖场景默认值）
    parser.add_argument(
        '--machine-fail-prob', type=float, default=None,
        help='机器故障概率（覆盖场景默认值）'
    )
    parser.add_argument(
        '--agv-fail-prob', type=float, default=None,
        help='AGV故障概率（覆盖场景默认值）'
    )
    parser.add_argument(
        '--job-add-prob', type=float, default=None,
        help='新增作业概率（覆盖场景默认值）'
    )
    parser.add_argument(
        '--machine-fail-duration-min', type=float, default=None,
        help='机器故障最短持续时间'
    )
    parser.add_argument(
        '--machine-fail-duration-max', type=float, default=None,
        help='机器故障最长持续时间'
    )
    parser.add_argument(
        '--agv-fail-duration-min', type=float, default=None,
        help='AGV故障最短持续时间'
    )
    parser.add_argument(
        '--agv-fail-duration-max', type=float, default=None,
        help='AGV故障最长持续时间'
    )

    # 日志级别
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别 (默认: INFO)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置日志
    level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO,
                 "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    logging.basicConfig(
        level=level_map[args.log_level],
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 从预设场景获取基础配置
    config = SCENARIOS[args.scenario].copy()
    description = config.pop("description", "")

    # 用 CLI 参数覆盖
    if args.machine_fail_prob is not None:
        config["machine_fail_prob"] = args.machine_fail_prob
    if args.agv_fail_prob is not None:
        config["agv_fail_prob"] = args.agv_fail_prob
    if args.job_add_prob is not None:
        config["job_add_prob"] = args.job_add_prob
    if args.machine_fail_duration_min is not None or args.machine_fail_duration_max is not None:
        dmin = args.machine_fail_duration_min or config["machine_fail_duration_range"][0]
        dmax = args.machine_fail_duration_max or config["machine_fail_duration_range"][1]
        config["machine_fail_duration_range"] = [dmin, dmax]
    if args.agv_fail_duration_min is not None or args.agv_fail_duration_max is not None:
        dmin = args.agv_fail_duration_min or config["agv_fail_duration_range"][0]
        dmax = args.agv_fail_duration_max or config["agv_fail_duration_range"][1]
        config["agv_fail_duration_range"] = [dmin, dmax]

    output_dir = Path(args.output_dir) if args.output_dir else None

    logger.info("=" * 60)
    logger.info("不确定性事件数据集生成")
    logger.info(f"场景: {args.scenario}" + (f" ({description})" if description else ""))
    logger.info(f"种子: {args.seed}")
    logger.info(f"配置: {json.dumps(config, indent=2)}")
    logger.info(f"数据集族: {args.families or '全部'}")
    logger.info("=" * 60)

    generate_all(
        scenario=args.scenario,
        config=config,
        seed=args.seed,
        families=args.families,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
