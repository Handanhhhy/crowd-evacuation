"""
设备信息打印工具

在所有训练脚本中使用，统一打印PyTorch/GPU设备信息。
"""

import torch
from typing import Optional


def print_device_info(header: str = "系统设备信息", show_details: bool = True):
    """打印设备信息
    
    Args:
        header: 标题
        show_details: 是否显示详细信息（GPU显存等）
    """
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA信息
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        try:
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            if show_details:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"\n  GPU {i}: {props.name}")
                    print(f"    显存: {props.total_memory / 1024**3:.2f} GB")
                    print(f"    计算能力: {props.major}.{props.minor}")
                    if torch.cuda.is_available():
                        print(f"    当前显存使用: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB / {props.total_memory / 1024**3:.2f} GB")
        except Exception as e:
            print(f"  (获取CUDA详细信息时出错: {e})")
    
    # MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS可用 (Apple Silicon): {mps_available}")
    
    if not cuda_available and not mps_available:
        print("  (将使用CPU进行训练)")
    
    print("=" * 60 + "\n")


def get_device(device: Optional[str] = None) -> str:
    """自动检测或返回指定设备
    
    Args:
        device: 设备名称 ('auto', 'cuda', 'mps', 'cpu')，None表示自动检测
        
    Returns:
        设备名称
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def print_device_selection(device: str):
    """打印设备选择信息
    
    Args:
        device: 选择的设备名称
    """
    print(f"\n使用设备: {device}")
    
    if device == "cuda" and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print(f"  Apple Silicon GPU (MPS)")
    else:
        print(f"  CPU")
    print()
