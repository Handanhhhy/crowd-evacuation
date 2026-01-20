"""
下载 ETH-UCY 行人轨迹数据集
"""

import os
import urllib.request
from pathlib import Path


def download_eth_ucy():
    """下载 ETH-UCY 数据集"""

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw" / "eth_ucy"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 使用另一个可用的数据源
    base_url = "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/master/train/real_data"

    datasets = {
        "biwi_eth": f"{base_url}/biwi_eth.ndjson",
        "biwi_hotel": f"{base_url}/biwi_hotel.ndjson",
        "crowds_zara01": f"{base_url}/crowds_zara01.ndjson",
        "crowds_zara02": f"{base_url}/crowds_zara02.ndjson",
    }

    print("=" * 50)
    print("下载 ETH-UCY 行人轨迹数据集")
    print("=" * 50)

    success_count = 0
    for name, url in datasets.items():
        output_path = data_dir / f"{name}.ndjson"

        if output_path.exists():
            print(f"[跳过] {name} 已存在")
            success_count += 1
            continue

        print(f"[下载] {name}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"  -> 保存到 {output_path}")
            success_count += 1
        except Exception as e:
            print(f"  [错误] {e}")

    if success_count == 0:
        print("\n外部下载失败，生成模拟数据用于开发测试...")
        generate_synthetic_data(data_dir)

    print(f"\n数据目录: {data_dir}")
    list_data_files(data_dir)


def generate_synthetic_data(data_dir):
    """生成合成轨迹数据（当外部数据不可用时）"""
    import numpy as np

    np.random.seed(42)

    # 生成类似ETH-UCY格式的数据
    # 格式: frame_id, ped_id, x, y
    data = []

    n_pedestrians = 100
    n_frames = 200

    for ped_id in range(n_pedestrians):
        # 随机起点
        start_x = np.random.uniform(0, 10)
        start_y = np.random.uniform(0, 10)

        # 随机终点
        end_x = np.random.uniform(20, 30)
        end_y = np.random.uniform(0, 10)

        # 随机起始帧
        start_frame = np.random.randint(0, 100)

        # 生成轨迹
        duration = np.random.randint(50, 150)
        for i in range(duration):
            frame = start_frame + i
            if frame >= n_frames:
                break

            # 线性插值 + 随机扰动
            t = i / duration
            x = start_x + t * (end_x - start_x) + np.random.normal(0, 0.1)
            y = start_y + t * (end_y - start_y) + np.random.normal(0, 0.1)

            data.append([frame, ped_id, x, y])

    # 按帧排序
    data = sorted(data, key=lambda x: (x[0], x[1]))

    # 保存
    output_path = data_dir / "synthetic_eth.txt"
    with open(output_path, 'w') as f:
        for row in data:
            f.write(f"{row[0]}\t{row[1]}\t{row[2]:.4f}\t{row[3]:.4f}\n")

    print(f"  -> 生成合成数据: {output_path}")
    print(f"  -> {len(data)} 条记录, {n_pedestrians} 个行人")


def list_data_files(data_dir):
    """列出数据文件"""
    print("\n数据文件:")
    for f in data_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name}: {size/1024:.1f} KB")


if __name__ == "__main__":
    download_eth_ucy()
