"""
自动化训练循环脚本

功能：
1. 启动后端（后台+训练模式）
2. 随机读取 dataset/agv-instances 下的数据文件，生成 pipeline_config.yaml
3. POST YAML_UPLOAD 注入配置
4. POST MAP_RENDER 启动工厂生成
5. 轮询 JOBS_PROGRESS 检测任务完成，完成后关闭后端
6. 重复 2-5 直到达到结束条件（reward 阈值、loss 阈值、或最大迭代次数）

收敛检测条件（可组合）：
- reward 阈值：当 episode reward >= 阈值时判定收敛
- loss 阈值：当所有 loss 指标 < 阈值时判定收敛
- 收敛耐心度：连续满足所有条件的迭代次数

用法：
    uv run python scripts/auto_train_loop.py [选项]

示例：
    # 使用默认端口 8000
    python scripts/auto_train_loop.py

    # 指定自定义端口
    python scripts/auto_train_loop.py --port 8001
    python scripts/auto_train_loop.py --port 9000 --log-level INFO

    # 组合多个参数
    python scripts/auto_train_loop.py --port 8080 --reward-threshold 500 --max-iterations 100

    # 启用 loss 收敛检测
    python scripts/auto_train_loop.py --loss-threshold 0.01 --patience 3

    # 同时使用 reward 和 loss 收敛检测（两者都满足时停止）
    python scripts/auto_train_loop.py --reward-threshold 500 --loss-threshold 0.1 --patience 2
"""
import subprocess
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
import requests
from requests import Session

# 禁用代理，直接连接本地后端
session = Session()
session.trust_env = False  # 忽略环境变量中的 HTTP_PROXY, HTTPS_PROXY 等

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# === 配置 ===
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = None  # 动态端口，可通过命令行指定
BACKEND_URL = None  # 将在 parse_args() 后初始化
DATA_DIR = Path(__file__).parent.parent / "dataset" / "agv-instances"
TRAINING_LOGS = Path(__file__).parent.parent / "training_logs"
RESULTS_DIR = TRAINING_LOGS / "results"

# 结束条件配置
REWARD_THRESHOLD = 1000.0  # 累积 reward 阈值
LOSS_THRESHOLD = None  # Loss 阈值（None 表示不检查）
CONVERGENCE_PATIENCE = 1  # 收敛满足条件的连续迭代次数
MAX_ITERATIONS = 1000  # 最大迭代次数（防止无限循环）
POLL_INTERVAL = 2  # 轮询间隔（秒）
POLL_TIMEOUT = 7200  # 单次训练超时（秒）= 2小时

# 请求重试配置
REQUEST_MAX_RETRIES = 3  # 最大重试次数
REQUEST_RETRY_DELAY = 1  # 重试间隔（秒）
REQUEST_SHORT_TIMEOUT = 5  # 短请求超时（秒）
REQUEST_MEDIUM_TIMEOUT = 15  # 中等请求超时（秒）
REQUEST_LONG_TIMEOUT = 30  # 长请求超时（秒）

# === API 端点 ===
API_FACTORY_SWITCH = f"{BACKEND_URL}/factory/control/switch"
API_HEALTH = f"{BACKEND_URL}/health"
API_YAML_UPLOAD = f"{BACKEND_URL}/{{config_name}}/yaml/upload"
API_MAP_RENDER = f"{BACKEND_URL}/map/render"
API_JOBS_PROGRESS = f"{BACKEND_URL}/jobs/progress"

# === 全局变量 ===
backend_process: Optional[subprocess.Popen] = None
# 标记是否已经初始化过后端（用于判断是否需要重启）
_backend_initialized: bool = False
# 日志记录器
logger = logging.getLogger("auto_train_loop")


def setup_logging(log_level: str = "INFO"):
    """
    配置日志系统
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
    """
    # 映射字符串到 logging 常量
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.setLevel(level)
    logger.info(f"日志级别已设置为: {log_level.upper()}")


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="自动化训练循环脚本（支持 reward/loss 收敛检测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/auto_train_loop.py
  python scripts/auto_train_loop.py --log-level DEBUG
  python scripts/auto_train_loop.py --log-level WARNING --backend-log-level ERROR
  python scripts/auto_train_loop.py --log-level WARNING --reward-threshold 500
  python scripts/auto_train_loop.py --max-iterations 100 --poll-interval 5
  python scripts/auto_train_loop.py --port 8001
  python scripts/auto_train_loop.py --port 9000 --log-level INFO
  python scripts/auto_train_loop.py --loss-threshold 0.01 --patience 3
  python scripts/auto_train_loop.py --reward-threshold 500 --loss-threshold 0.1 --patience 2
        """
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
        help='后端服务日志级别 (默认: INFO)'
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
        '--max-iterations',
        type=int,
        default=MAX_ITERATIONS,
        help=f'最大迭代次数 (默认: {MAX_ITERATIONS})'
    )
    
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=POLL_INTERVAL,
        help=f'轮询间隔（秒）(默认: {POLL_INTERVAL})'
    )
    
    parser.add_argument(
        '--poll-timeout',
        type=int,
        default=POLL_TIMEOUT,
        help=f'单次训练超时（秒）(默认: {POLL_TIMEOUT})'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='后端服务端口号 (默认: 8000)'
    )
    
    return parser.parse_args()


def retry_request(func, *args, max_retries: int = REQUEST_MAX_RETRIES, **kwargs) -> Optional[requests.Response]:
    """
    带重试机制的请求函数
    
    Args:
        func: requests 请求函数 (session.get, session.post 等)
        *args: 位置参数
        max_retries: 最大重试次数
        **kwargs: 关键字参数
        
    Returns:
        Optional[requests.Response]: 响应对象，失败返回 None
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = func(*args, **kwargs)
            if resp.status_code == 200:
                return resp
            else:
                logger.warning(f"请求失败 (尝试 {attempt}/{max_retries}): HTTP {resp.status_code}")
                if attempt < max_retries:
                    time.sleep(REQUEST_RETRY_DELAY)
                continue
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning(f"请求异常 (尝试 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(REQUEST_RETRY_DELAY)
            continue
    
    logger.error(f"请求失败，已重试 {max_retries} 次")
    return None


def reset_factory_state() -> bool:
    """
    重置工厂状态（清理之前的训练状态，准备下一次训练）

    通过调用 /factory/control/reset 端点重置后端核心。
    如果失败，会重启后端。

    Returns:
        bool: 重置是否成功
    """
    logger.info("重置工厂状态...")
    resp = retry_request(
        session.post,
        f"{BACKEND_URL}/factory/control/reset",
        timeout=REQUEST_MEDIUM_TIMEOUT
    )
    
    if resp and resp.status_code == 200:
        logger.info("工厂状态已重置")
        return True
    else:
        logger.warning("重置请求失败")

    # 重置失败，重启后端
    logger.info("重置失败，尝试重启后端...")
    return restart_backend()


def restart_backend() -> bool:
    """
    重启后端服务

    Returns:
        bool: 重启是否成功
    """
    global backend_process

    logger.info("重启后端服务...")
    stop_backend()

    if not start_backend():
        return False

    # 重新初始化 packet_factory
    logger.info("重新初始化 packet_factory...")
    resp = retry_request(
        session.post,
        API_FACTORY_SWITCH,
        json={"factory_id": "packet_factory"},
        timeout=REQUEST_LONG_TIMEOUT
    )
    
    if resp and resp.status_code == 200:
        data = resp.json()
        if data.get("status") == "ok":
            logger.info("packet_factory 重新初始化完成")
            return True
    
    logger.error("重新初始化失败")
    return False


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


def generate_pipeline_config(parsed_data: dict, config_name: str) -> str:
    """
    从解析的数据生成完整的 pipeline_config.yaml

    Args:
        parsed_data: 解析后的数据
        config_name: 配置名称

    Returns:
        str: YAML 格式的配置文件内容
    """
    # 生成 points
    points = [
        {
            "point": {
                "id": point_id,
                "coordinate": [x, y]
            }
        }
        for point_id, x, y in parsed_data['points']
    ]

    # 生成 links
    links = [
        {
            "link": {
                "id": link_id,
                "begin": point1_id,
                "end": point2_id
            }
        }
        for link_id, point1_id, point2_id, weight in parsed_data['links']
    ]

    # 生成 machines
    machines = [
        {
            "machine": {
                "id": machine_id,
                "type": "packet_factory.Machine",
                "point_id": point_id
            }
        }
        for machine_id, point_id in parsed_data['machines']
    ]

    # 生成 agvs
    agvs = [
        {
            "agv": {
                "id": agv_id,
                "type": "packet_factory.Agv",
                "point_id": point_id,
                "velocity": velocity,
                "capacity": 12
            }
        }
        for agv_id, point_id, velocity in parsed_data['agvs']
    ]

    # 计算地图尺寸
    all_x = [p[1] for p in parsed_data['points']]
    all_y = [p[2] for p in parsed_data['points']]
    width = int(max(all_x) + 5) if all_x else 20
    height = int(max(all_y) + 5) if all_y else 30

    # 生成 job_config
    jobs_yaml = []
    for job_id, operations in parsed_data["jobs"]:
        job_entry = {
            "job": {
                "id": job_id,
                "operations": []
            }
        }
        for op_idx, machine_options in enumerate(operations):
            op_entry = {
                "operation": {
                    "id": op_idx,
                    "machines": [
                        {"id": m, "time": d}
                        for m, d in machine_options
                    ]
                }
            }
            job_entry["job"]["operations"].append(op_entry)
        jobs_yaml.append(job_entry)

    # 组装完整的 pipeline_config
    pipeline_config = {
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
        "job_config": {
            "jobs": jobs_yaml
        },
        "map_config": {
            "width": width,
            "height": height,
            "points": points,
            "machines": machines,
            "links": links,
            "agvs": agvs
        }
    }

    return yaml.dump(pipeline_config, allow_unicode=True, sort_keys=False)


def start_backend(backend_log_level: str = "INFO") -> bool:
    """
    启动后端服务（后台模式）

    Args:
        backend_log_level: 后端日志级别 (DEBUG, INFO, WARNING, ERROR)

    Returns:
        bool: 启动是否成功
    """
    global backend_process

    backend_dir = Path(__file__).parent.parent / "application" / "backend"
    project_root = backend_dir.parent.parent

    logger.info("启动后端服务...")
    logger.debug(f"工作目录: {project_root}")
    logger.info(f"后端日志级别: {backend_log_level.upper()}")

    try:
        # 设置环境变量，传递给后端进程
        env = os.environ.copy()
        env['BACKEND_LOG_LEVEL'] = backend_log_level.upper()
        
        # 确保 PYTHONPATH 包含项目根目录
        env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')

        # 启动 uvicorn，从项目根目录运行
        backend_process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "application.backend.server:app",
                "--host", BACKEND_HOST,
                "--port", str(BACKEND_PORT),
                "--log-level", "info"
            ],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # 等待后端就绪，最多等待60秒
        logger.info("等待后端就绪...")
        if wait_for_backend(timeout=60):
            logger.info("后端服务已启动 ✅")
            return True
        else:
            # 读取错误信息
            logger.error("后端服务启动超时 ❌")
            # 检查进程状态
            if backend_process.poll() is not None:
                _, stderr = backend_process.communicate(timeout=5)
                logger.error(f"后端进程已退出，stderr: {stderr[:500] if stderr else 'N/A'}")
            else:
                logger.error("后端进程仍在运行但未响应")
            stop_backend()
            return False

    except Exception as e:
        logger.error(f"启动后端失败: {e}")
        return False


def wait_for_backend(timeout: int = 60) -> bool:
    """
    等待后端就绪

    Args:
        timeout: 超时时间（秒）

    Returns:
        bool: 是否就绪
    """
    import time
    # 等待2秒让服务器启动
    time.sleep(2)
    start_time = time.time()
    while time.time() - start_time < timeout:
        resp = retry_request(
            session.get,
            API_HEALTH,
            timeout=REQUEST_SHORT_TIMEOUT,
            max_retries=1  # 健康检查只重试1次，快速失败
        )
        if resp and resp.status_code == 200:
            return True
        time.sleep(1)
    return False


def stop_backend():
    """关闭后端服务"""
    global backend_process

    if backend_process:
        logger.info("关闭后端服务...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend_process.kill()
            backend_process.wait()
        backend_process = None
        logger.info("后端服务已关闭 ✅")


def upload_config(config_name: str, yaml_content: str) -> bool:
    """
    上传 YAML 配置到后端

    Args:
        config_name: 配置名称
        yaml_content: YAML 文件内容

    Returns:
        bool: 上传是否成功
    """
    files = {'file': (f'{config_name}.yaml', yaml_content, 'text/plain')}
    # config_name 作为 query 参数传递
    resp = retry_request(
        session.post,
        f"{BACKEND_URL}/yaml/upload?config_name={config_name}",
        files=files,
        timeout=REQUEST_MEDIUM_TIMEOUT
    )
    
    if resp and resp.status_code == 200:
        logger.info(f"配置 {config_name} 上传成功")
        return True
    else:
        logger.error("配置上传失败")
        return False


def render_factory(config_name: str) -> bool:
    """
    启动工厂渲染（异步方式）

    后端通过线程池异步处理训练，API 会立即返回。
    脚本会在 poll_jobs_completion 中轮询训练状态。

    Args:
        config_name: 配置名称

    Returns:
        bool: 请求是否发送成功
    """
    import threading

    def send_render_request():
        """后台发送渲染请求"""
        resp = retry_request(
            session.post,
            API_MAP_RENDER,
            json={"target_factory": config_name},
            timeout=REQUEST_LONG_TIMEOUT
        )
        
        if resp and resp.status_code == 200:
            logger.info(f"工厂 {config_name} 渲染已启动")
        else:
            logger.error("工厂渲染启动失败")

    try:
        # 在后台线程中发送请求，避免阻塞
        thread = threading.Thread(target=send_render_request, daemon=True)
        thread.start()
        return True
    except Exception as e:
        logger.error(f"启动渲染线程失败: {e}")
        return False


def wait_for_env_ready(timeout: int = 30) -> bool:
    """
    等待环境准备好（之前的训练完全结束）

    Args:
        timeout: 超时时间（秒）

    Returns:
        bool: 是否准备好
    """
    import time
    start_time = time.time()
    while time.time() - start_time < timeout:
        resp = retry_request(
            session.get,
            f"{BACKEND_URL}/factory/alive",
            timeout=REQUEST_SHORT_TIMEOUT,
            max_retries=1
        )
        
        if resp and resp.status_code == 200:
            data = resp.json()
            # 如果 is_alive 为 False，说明环境已关闭，可以继续
            if not data.get('is_alive', True):
                return True
        
        time.sleep(0.5)
    
    # 超时也算准备好（可能有其他问题）
    return True


def poll_jobs_completion(timeout: int = POLL_TIMEOUT) -> Tuple[bool, List[dict], float]:
    """
    轮询任务完成状态

    通过检查 /factory/alive 端点来判断训练是否完成。
    支持两种完成检测方式：
    1. is_alive 从 True 变为 False（训练正常完成）
    2. training_completed=True（训练快速完成，客户端没来得及看到 is_alive=True）

    Args:
        timeout: 超时时间（秒）

    Returns:
        Tuple[bool, List[dict], float]: (是否完成, 任务列表, makespan)
    """
    start_time = time.time()
    makespan = 0.0
    was_alive = False

    # 开始轮询：使用 /factory/alive 判断训练是否完成
    logger.info("开始监控任务进度...")
    last_status_log = 0
    while time.time() - start_time < timeout:
        resp = retry_request(
            session.get,
            f"{BACKEND_URL}/factory/alive",
            timeout=REQUEST_MEDIUM_TIMEOUT,
            max_retries=2
        )
        
        if resp and resp.status_code == 200:
            data = resp.json()
            is_alive = data.get('is_alive', False)
            training_completed = data.get('training_completed', False)
            makespan = float(data.get('makespan', 0))

            # 检测训练完成：training_completed=True
            if training_completed:
                logger.info(f"训练完成，检测到 training_completed=True (makespan: {makespan:.2f}s)")

                # 获取最终的任务状态
                resp_jobs = retry_request(
                    session.get,
                    API_JOBS_PROGRESS,
                    timeout=REQUEST_MEDIUM_TIMEOUT,
                    max_retries=2
                )
                
                if resp_jobs and resp_jobs.status_code == 200:
                    jobs_data = resp_jobs.json().get('jobs', [])
                    return True, jobs_data, makespan

                return True, [], makespan

            # 检测 is_alive 从 True 变为 False
            if was_alive and not is_alive:
                elapsed = time.time() - start_time
                logger.info(f"训练完成，检测到环境已结束 (elapsed: {elapsed:.2f}s)")

                # 获取最终的任务状态
                resp_jobs = retry_request(
                    session.get,
                    API_JOBS_PROGRESS,
                    timeout=REQUEST_MEDIUM_TIMEOUT,
                    max_retries=2
                )
                
                if resp_jobs and resp_jobs.status_code == 200:
                    jobs_data = resp_jobs.json().get('jobs', [])
                    return True, jobs_data, elapsed

                return True, [], elapsed

            was_alive = is_alive

            # 每30秒打印一次状态
            if time.time() - last_status_log > 30:
                elapsed = time.time() - start_time
                # 从 /factory/alive 响应获取实时训练指标
                live_metrics = data.get('training_metrics', {}) if resp and resp.status_code == 200 else {}
                if not live_metrics:
                    live_metrics = get_latest_training_metrics()
                metric_str = ""
                if live_metrics:
                    ep_reward = live_metrics.get('episode_reward', None)
                    epsilon = live_metrics.get('epsilon', None)
                    if ep_reward is not None:
                        metric_str += f", reward={ep_reward:.4f}"
                    if epsilon is not None:
                        metric_str += f", epsilon={epsilon}"
                    loss_info = {k: f"{v:.6f}" for k, v in live_metrics.items()
                                if 'loss' in k.lower() and isinstance(v, (int, float))}
                    if loss_info:
                        metric_str += f", {loss_info}"
                logger.info(f"训练进行中... ({int(elapsed)}s{metric_str})")
                last_status_log = time.time()

        time.sleep(POLL_INTERVAL)

    # 超时，获取最终状态
    elapsed = time.time() - start_time
    resp_jobs = retry_request(
        session.get,
        API_JOBS_PROGRESS,
        timeout=REQUEST_MEDIUM_TIMEOUT,
        max_retries=2
    )
    
    if resp_jobs and resp_jobs.status_code == 200:
        jobs_data = resp_jobs.json().get('jobs', [])
        return False, jobs_data, elapsed
    
    return False, [], elapsed


def get_latest_training_result() -> Optional[dict]:
    """
    获取最新的训练结果

    Returns:
        Optional[dict]: 训练结果字典，如果不存在则返回 None
    """
    if not RESULTS_DIR.exists():
        return None

    # 查找最新的结果目录
    result_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    if not result_dirs:
        return None

    latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
    report_file = latest_dir / "training_report.json"

    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None


def get_actual_makespan_from_result() -> float:
    """
    从最新的训练结果文件获取实际的 makespan

    Returns:
        float: 实际的 makespan 值，如果不存在则返回 0.0
    """
    result = get_latest_training_result()
    if result and 'makespan' in result:
        return float(result['makespan'])
    return 0.0


def get_cumulative_reward() -> float:
    """
    获取最新一次训练的 episode reward

    优先从 training_report.json 的 training_metrics.episode_reward 读取，
    其次尝试从 training_metrics.decision_stats 中读取，
    最后尝试从 agent 模型文件中读取 training_history。

    Returns:
        float: episode reward 值
    """
    result = get_latest_training_result()

    # 优先从 training_metrics 读取
    if result and 'training_metrics' in result:
        metrics = result['training_metrics']
        if 'episode_reward' in metrics:
            return float(metrics['episode_reward'])

    # 尝试从 decision_stats 读取（兼容旧格式）
    if result and 'decision_stats' in result:
        stats = result['decision_stats']
        for key in ['total_reward', 'episode_reward', 'total_rewards', 'cumulative_reward', 'reward']:
            if key in stats:
                return float(stats[key])

    # 最后尝试从 agent 模型目录中读取 training_history
    models_dir = TRAINING_LOGS / "models"
    if models_dir.exists():
        for agent_dir in sorted(models_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
            if not agent_dir.is_dir():
                continue
            # 尝试 JSON 格式
            json_file = agent_dir / "agent_model.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        model_data = json.load(f)
                    history = model_data.get('training_history', {})
                    for key in ['total_rewards', 'episode_reward']:
                        vals = history.get(key, [])
                        if vals:
                            return float(vals[-1]) if isinstance(vals[-1], (int, float)) else float(sum(vals))
                except Exception:
                    pass

            # 尝试 PyTorch 格式（需读取 JSON backup 或跳过）
            pt_file = agent_dir / "agent_model.pt"
            if pt_file.exists():
                try:
                    import torch
                    checkpoint = torch.load(str(pt_file), map_location='cpu', weights_only=False)
                    history = checkpoint.get('training_history', {})
                    for key in ['total_rewards', 'episode_reward']:
                        vals = history.get(key, [])
                        if vals:
                            return float(vals[-1]) if isinstance(vals[-1], (int, float)) else float(sum(vals))
                except Exception:
                    pass

    return 0.0


def get_latest_training_metrics() -> Dict:
    """
    获取最新训练结果的 training_metrics 字典

    Returns:
        Dict: training_metrics 字典，如果不存在则返回空字典
    """
    result = get_latest_training_result()
    if result and 'training_metrics' in result:
        return result['training_metrics']
    return {}


def check_convergence(iteration_metrics: List[Dict], reward_threshold: float,
                      loss_threshold: Optional[float], patience: int) -> Tuple[bool, str]:
    """
    检查训练是否收敛

    收敛条件：
    1. episode_reward >= reward_threshold（如果 reward_threshold > 0）
    2. 所有 loss 指标 < loss_threshold（如果 loss_threshold 不为 None）
    3. 以上条件连续满足 patience 次迭代

    Args:
        iteration_metrics: 历次迭代的 metrics 列表
        reward_threshold: reward 阈值
        loss_threshold: loss 阈值（None 表示不检查）
        patience: 连续满足条件的迭代次数

    Returns:
        Tuple[bool, str]: (是否收敛, 收敛原因描述)
    """
    if not iteration_metrics:
        return False, ""

    # 检查最近 patience 次迭代
    recent = iteration_metrics[-patience:] if len(iteration_metrics) >= patience else iteration_metrics
    if len(recent) < patience:
        return False, ""

    # 所有 recent 迭代都需要满足条件
    all_reward_ok = True
    all_loss_ok = True
    loss_details = []

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

    reasons = []
    if all_reward_ok and reward_threshold > 0:
        reasons.append(f"reward >= {reward_threshold}")
    if all_loss_ok and loss_threshold is not None:
        reasons.append(f"all losses < {loss_threshold}")

    # 所有启用的条件都满足才收敛
    if reward_threshold > 0 and loss_threshold is not None:
        converged = all_reward_ok and all_loss_ok
    elif reward_threshold > 0:
        converged = all_reward_ok
    elif loss_threshold is not None:
        converged = all_loss_ok
    else:
        converged = False

    if converged:
        reason = ", ".join(reasons)
        return True, f"连续 {patience} 次满足: {reason}"

    return False, ""


def check_reward_threshold() -> bool:
    """
    检查累积 reward 是否达到阈值（兼容旧接口）

    Returns:
        bool: 是否达到阈值
    """
    cumulative_reward = get_cumulative_reward()
    logger.info(f"当前 episode reward: {cumulative_reward:.2f}, 阈值: {REWARD_THRESHOLD:.2f}")
    return cumulative_reward >= REWARD_THRESHOLD


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


def switch_to_packet_factory() -> bool:
    """
    切换到 packet_factory 工厂（注册后端路由）

    Returns:
        bool: 切换是否成功
    """
    resp = retry_request(
        session.post,
        API_FACTORY_SWITCH,
        json={"factory_id": "packet_factory"},
        timeout=REQUEST_MEDIUM_TIMEOUT
    )
    
    if resp and resp.status_code == 200:
        data = resp.json()
        if data.get("status") == "ok":
            logger.info("已切换到 packet_factory")
            return True
        else:
            logger.error(f"切换失败: {data}")
            return False
    else:
        logger.error("切换请求失败")
        return False


def run_training_iteration(iteration: int, data_file: Path) -> Tuple[bool, float]:
    """
    执行一次训练迭代

    Args:
        iteration: 当前迭代编号
        data_file: 数据文件路径

    Returns:
        Tuple[bool, float]: (是否成功, makespan)
    """
    # 显示相对于 DATA_DIR 的路径
    relative_path = data_file.relative_to(DATA_DIR)
    config_name = f"auto_train_{iteration}_{data_file.stem}"

    logger.info("\n" + "="*60)
    logger.info(f"[迭代 {iteration}] 使用数据文件: {relative_path}")
    logger.info(f"{'='*60}")

    try:
        # 1. 解析数据文件
        logger.info("[INFO] 解析数据文件...")
        parsed_data = parse_agv_instance(data_file)
        logger.info(f"[INFO] 解析完成: {parsed_data['job_count']} jobs, "
              f"{parsed_data['machine_count']} machines, "
              f"{parsed_data['agv_count']} AGVs")

        # 2. 生成配置
        logger.info("[INFO] 生成 pipeline_config...")
        yaml_content = generate_pipeline_config(parsed_data, config_name)

        # 3. 上传配置
        logger.info("[INFO] 上传配置...")
        if not upload_config(config_name, yaml_content):
            return False, 0.0

        # 4. 启动渲染
        logger.info("[INFO] 启动工厂渲染...")
        if not render_factory(config_name):
            return False, 0.0

        # 5. 轮询完成
        logger.info("[INFO] 等待训练完成...")
        success, jobs, elapsed_time = poll_jobs_completion()

        # 获取实际的 makespan（从训练结果文件）
        actual_makespan = get_actual_makespan_from_result()

        if success:
            if actual_makespan > 0:
                logger.info(f"[INFO] 训练完成! Makespan: {actual_makespan:.2f}s")
            else:
                logger.info(f"[INFO] 训练完成! Makespan: {elapsed_time:.2f}s (估计)")
        else:
            if actual_makespan > 0:
                logger.warning(f"[WARN] 训练超时! Makespan: {actual_makespan:.2f}s")
            else:
                logger.warning(f"[WARN] 训练超时! Makespan: {elapsed_time:.2f}s (估计)")

        # 训练完成后，等待一段时间让线程完全结束
        logger.info("[INFO] 等待线程完全结束...")
        time.sleep(3)

        return success, actual_makespan if actual_makespan > 0 else elapsed_time

    except Exception as e:
        logger.error(f"[ERROR] 训练迭代失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0


def signal_handler(sig, frame):
    """信号处理器"""
    logger.info("\n收到信号，准备退出...")
    stop_backend()
    sys.exit(0)


def main():
    """主函数"""
    global backend_process, BACKEND_PORT, BACKEND_URL

    # 解析命令行参数
    args = parse_args()

    # 更新全局配置
    global REWARD_THRESHOLD, MAX_ITERATIONS, POLL_INTERVAL, POLL_TIMEOUT
    global LOSS_THRESHOLD, CONVERGENCE_PATIENCE
    BACKEND_PORT = args.port
    BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
    REWARD_THRESHOLD = args.reward_threshold
    MAX_ITERATIONS = args.max_iterations
    POLL_INTERVAL = args.poll_interval
    POLL_TIMEOUT = args.poll_timeout
    LOSS_THRESHOLD = args.loss_threshold
    CONVERGENCE_PATIENCE = args.patience

    # 重新计算 API 端点（因为 BACKEND_URL 已更新）
    global API_FACTORY_SWITCH, API_HEALTH, API_YAML_UPLOAD, API_MAP_RENDER, API_JOBS_PROGRESS
    API_FACTORY_SWITCH = f"{BACKEND_URL}/factory/control/switch"
    API_HEALTH = f"{BACKEND_URL}/health"
    API_YAML_UPLOAD = f"{BACKEND_URL}/{{config_name}}/yaml/upload"
    API_MAP_RENDER = f"{BACKEND_URL}/map/render"
    API_JOBS_PROGRESS = f"{BACKEND_URL}/jobs/progress"

    # 设置前端脚本日志级别
    setup_logging(args.log_level)

    logger.info("="*60)
    logger.info("自动化训练循环脚本")
    logger.info(f"后端端口: {BACKEND_PORT}")
    logger.info(f"Reward 阈值: {REWARD_THRESHOLD:.2f}")
    logger.info(f"Loss 阈值: {LOSS_THRESHOLD if LOSS_THRESHOLD is not None else '不检查'}")
    logger.info(f"收敛耐心度: {CONVERGENCE_PATIENCE}")
    logger.info(f"最大迭代次数: {MAX_ITERATIONS}")
    logger.info(f"前端脚本日志级别: {args.log_level.upper()}")
    logger.info(f"后端服务日志级别: {args.backend_log_level.upper()}")
    logger.info("="*60)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 跨迭代指标追踪
    iteration_metrics: List[Dict] = []

    try:
        # 训练循环
        iteration = 0
        while iteration < MAX_ITERATIONS:
            # 检查是否达到收敛条件
            if iteration > 0 and iteration_metrics:
                converged, reason = check_convergence(
                    iteration_metrics, REWARD_THRESHOLD, LOSS_THRESHOLD, CONVERGENCE_PATIENCE
                )
                if converged:
                    logger.info(f"\n=== 训练收敛，停止迭代 ===")
                    logger.info(f"收敛原因: {reason}")
                    break

            # 随机选择数据文件
            data_file = select_random_data_file()
            if not data_file:
                logger.error("未找到数据文件")
                break

            # 每次迭代开始时重启后端
            logger.info("="*60)
            logger.info(f"[迭代 {iteration}] 启动新的后端实例...")
            logger.info("="*60)

            # 如果后端已在运行，先关闭
            if backend_process is not None:
                logger.info("关闭上一次迭代的后端...")
                stop_backend()
                time.sleep(2)  # 等待端口释放

            # 启动新的后端（传递后端日志级别）
            if not start_backend(backend_log_level=args.backend_log_level):
                logger.error("后端启动失败，跳过本次迭代")
                iteration += 1
                continue

            # 切换到 packet_factory
            logger.info("切换到 packet_factory...")
            if not switch_to_packet_factory():
                logger.error("工厂切换失败，跳过本次迭代")
                stop_backend()
                iteration += 1
                continue

            # 执行训练
            success, makespan = run_training_iteration(iteration, data_file)

            # 收集本次迭代的指标
            metrics = get_latest_training_metrics()
            metrics['makespan'] = makespan
            metrics['iteration_success'] = success
            iteration_metrics.append(metrics)

            # 输出本次迭代的关键指标
            episode_reward = metrics.get('episode_reward', 0.0)
            epsilon = metrics.get('epsilon', 'N/A')
            loss_keys = {k: v for k, v in metrics.items() if 'loss' in k.lower() and isinstance(v, (int, float))}
            logger.info(f"[迭代 {iteration} 指标] reward={episode_reward:.4f}, "
                        f"makespan={makespan:.2f}, epsilon={epsilon}"
                        + (f", losses={{{', '.join(f'{k}={v:.6f}' for k, v in loss_keys.items())}}}"
                           if loss_keys else ""))

            iteration += 1

            # 短暂休息，避免太快
            time.sleep(2)

        if iteration >= MAX_ITERATIONS:
            logger.warning(f"\n达到最大迭代次数 {MAX_ITERATIONS}")

        # 打印最终统计
        cumulative_reward = get_cumulative_reward()
        final_metrics = iteration_metrics[-1] if iteration_metrics else {}
        final_reward = final_metrics.get('episode_reward', 0.0)
        final_losses = {k: v for k, v in final_metrics.items() if 'loss' in k.lower() and isinstance(v, (int, float))}

        logger.info("\n" + "="*60)
        logger.info("训练完成!")
        logger.info(f"总迭代次数: {iteration}")
        logger.info(f"最终 episode reward: {final_reward:.4f}")
        if final_losses:
            logger.info(f"最终 losses: {{{', '.join(f'{k}={v:.6f}' for k, v in final_losses.items())}}}")
        logger.info(f"最终 makespan: {final_metrics.get('makespan', 0.0):.2f}")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.info("\n收到键盘中断信号")
    finally:
        # 确保后端被关闭
        stop_backend()
        logger.info("="*60)


if __name__ == "__main__":
    main()
