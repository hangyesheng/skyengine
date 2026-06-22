"""
PyTorch 设备解析工具

提供 resolve_device() / log_device()，统一处理训练 Agent 的设备选择。

背景：
    在存在异常 GPU（例如 nvidia-smi 报 "Unable to determine the device handle"
    的坏卡）的服务器上，torch.cuda.is_available() 可能返回 True，
    但真正执行 torch._C._cuda_init() 时会枚举到坏卡并抛
    RuntimeError: CUDA unknown error，导致整个训练流程崩溃。

    resolve_device() 通过「主动触发 CUDA 初始化 + 微小张量分配」提前验证，
    任何环节失败均自动回退到 CPU，保证训练可在有坏卡的服务器上继续运行
    （若要用 GPU 加速，需配合 CUDA_VISIBLE_DEVICES 隐藏坏卡）。
"""
import torch
from typing import Optional, Union

from executor.packet_factory.logger.logger import LOGGER


def _cuda_works(device_index: int = 0) -> None:
    """
    探测 CUDA 是否真正可用：触发 _cuda_init 并分配一个微小张量。

    任何一步抛异常都说明 CUDA 不可用（坏卡 / 驱动问题 / 环境变量错误等），
    由调用方捕获后回退 CPU。
    """
    torch.cuda.get_device_name(device_index)        # 触发 _lazy_init / _cuda_init
    torch.zeros(1, device=f'cuda:{device_index}')   # 进一步验证显存分配


def resolve_device(device: Optional[Union[str, torch.device]] = None,
                   tag: str = "Agent") -> torch.device:
    """
    解析 torch 计算设备，CUDA 初始化失败时自动回退到 CPU。

    Args:
        device: 'auto'/None 自动选择；'cuda'/'cuda:0'/'cpu' 等显式指定
        tag: 日志前缀（如 Agent 名称）

    Returns:
        torch.device：可用的计算设备（cuda 或 cpu）
    """
    # 显式指定且非 auto：尝试使用，初始化失败则回退 CPU
    if device is not None and device != 'auto':
        try:
            dev = torch.device(device)
            if dev.type == 'cuda':
                idx = dev.index if dev.index is not None else 0
                _cuda_works(idx)
            return dev
        except Exception as e:
            LOGGER.warning(f"[{tag}] 指定设备 '{device}' 不可用，回退到 CPU: {e}")
            return torch.device('cpu')

    # auto / None：优先 CUDA，初始化失败回退 CPU
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _cuda_works(0)
            return torch.device('cuda')
    except Exception as e:
        LOGGER.warning(f"[{tag}] CUDA 不可用，回退到 CPU: {e}")
    return torch.device('cpu')


def log_device(device: torch.device, tag: str = "Agent"):
    """打印当前使用的计算设备"""
    if device.type == 'cuda':
        try:
            name = torch.cuda.get_device_name(device if device.index is not None else 0)
        except Exception:
            name = "CUDA"
        LOGGER.info(f"[{tag}] Using CUDA: {name}")
    else:
        LOGGER.info(f"[{tag}] Using CPU")
