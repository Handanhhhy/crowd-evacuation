"""配置文件加载工具"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载YAML配置文件

    Args:
        config_path: 配置文件路径，默认使用 configs/default.yaml

    Returns:
        配置字典
    """
    if config_path is None:
        # 默认配置路径
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "default.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """保存配置到YAML文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
