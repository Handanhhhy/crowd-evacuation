"""
环境测试脚本
运行此脚本检查所有依赖是否正确安装
"""

def test_imports():
    print("检查依赖...")
    print("-" * 40)

    modules = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("torch", "PyTorch"),
        ("sklearn", "Scikit-learn"),
        ("xgboost", "XGBoost"),
        ("pandas", "Pandas"),
        ("yaml", "PyYAML"),
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable-Baselines3"),
    ]

    success = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [X] {name} - {e}")
            success = False

    # 单独检查 pysocialforce
    print("-" * 40)
    print("检查 PySocialForce...")
    try:
        import pysocialforce
        print(f"  [OK] PySocialForce (可直接使用官方库)")
    except ImportError:
        print(f"  [!] PySocialForce 未安装，将使用自定义实现")

    print("-" * 40)

    # 检查 PyTorch cuda
    print("检查 PyTorch cuda 支持...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA 可用: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print(f"  [OK] MPS 可用 (Apple Silicon)")
        else:
            print(f"  [!] 仅 cuda 模式（训练会较慢，但可以运行）")
    except Exception as e:
        print(f"  [X] 检查失败: {e}")

    print("-" * 40)

    if success:
        print("\n所有核心依赖检查通过！")
        print("运行: python examples/demo_sfm.py 开始演示")
    else:
        print("\n部分依赖缺失，请运行:")
        print("  pip install -r requirements.txt")

    return success


if __name__ == "__main__":
    test_imports()
