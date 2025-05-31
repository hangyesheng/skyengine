'''
@Project ：tiangong 
@File    ：load_config.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/5/31 1:30 
'''
# config/loader.py

import yaml
import os

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if "config" not in raw_config:
        raise ValueError("Missing 'config' section in configuration.")

    sky_config = raw_config["config"]

    return sky_config
