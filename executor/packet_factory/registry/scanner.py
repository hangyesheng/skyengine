'''
@Project ：tiangong 
@File    ：scanner.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/5/31 0:38 
'''
import importlib
import pkgutil
import executor.packet_factory
import time
import logging

import yaml
import os
from executor.packet_factory.registry.registry import component_registry
from pathlib import Path


def load_config(config: str | dict):
    if isinstance(config, str):
        if not os.path.exists(config):
            raise FileNotFoundError(f"Configuration file not found: {config}")

        with open(config, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if "config" not in raw_config:
            raise ValueError("Missing 'config' section in configuration.")

        # 获取文件内的配置
        sky_config = raw_config["config"]
        # 保存配置文件的路径
        sky_config['config_path'] = Path(config)
        component_registry['config'] = sky_config
    elif isinstance(config, dict):
        # 保存配置文件的路径
        component_registry['config'] = config


def scan_and_register_components():
    """
    自动导入并触发装饰器注册
    
    优化说明：
    1. 添加超时保护（60秒）
    2. 添加详细日志，便于定位卡在哪个模块
    3. 添加异常捕获，单个模块失败不影响其他模块
    4. 性能监控，记录每个模块的导入时间
    """
    logger = logging.getLogger("auto_train_loop")
    
    package = executor.packet_factory
    module_list = list(pkgutil.walk_packages(package.__path__, package.__name__ + "."))
    
    total_modules = len(module_list)
    logger.info(f"[Scanner] Found {total_modules} modules to scan")
    
    start_time = time.time()
    max_scan_time = 60  # 最大扫描时间60秒
    success_count = 0
    failed_count = 0
    failed_modules = []
    
    for idx, (_, module_name, _) in enumerate(module_list, 1):
        # 检查是否超时
        elapsed = time.time() - start_time
        if elapsed > max_scan_time:
            logger.error(f"[Scanner] Scan timeout after {elapsed:.2f}s! Stopped at module {module_name}")
            logger.error(f"[Scanner] Progress: {idx}/{total_modules} modules scanned")
            break
        
        # 每10个模块打印一次进度
        if idx % 10 == 0 or idx == total_modules:
            logger.debug(f"[Scanner] Progress: {idx}/{total_modules} modules ({elapsed:.2f}s)")
        
        try:
            module_start = time.time()
            importlib.import_module(module_name)
            module_elapsed = time.time() - module_start
            
            # 如果单个模块导入超过2秒，记录警告
            if module_elapsed > 2.0:
                logger.warning(f"[Scanner] Slow module import: {module_name} took {module_elapsed:.2f}s")
            
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            failed_modules.append((module_name, str(e)))
            logger.warning(f"[Scanner] Failed to import module {module_name}: {e}")
    
    total_elapsed = time.time() - start_time
    logger.info(f"[Scanner] Component scan completed:")
    logger.info(f"  - Total modules: {total_modules}")
    logger.info(f"  - Success: {success_count}")
    logger.info(f"  - Failed: {failed_count}")
    logger.info(f"  - Time elapsed: {total_elapsed:.2f}s")
    
    if failed_modules:
        logger.warning(f"[Scanner] Failed modules:")
        for module_name, error in failed_modules[:10]:  # 只显示前10个失败的
            logger.warning(f"    - {module_name}: {error}")
        if len(failed_modules) > 10:
            logger.warning(f"    ... and {len(failed_modules) - 10} more")
