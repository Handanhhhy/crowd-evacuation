#!/usr/bin/env python
"""
下载人群疏散相关的公开数据集

支持的数据集：
1. Jülich Pedestrian Dynamics Data Archive - 最权威的疏散实验数据
2. Lyon Dense Crowd Dataset (2024-2025) - 高密度人群数据
3. Grand Central Station Dataset - 火车站场景
4. ETH-UCY Dataset - 经典基准数据集

使用方法:
    python scripts/download_evacuation_datasets.py --dataset juelich
    python scripts/download_evacuation_datasets.py --dataset all
    python scripts/download_evacuation_datasets.py --dataset juelich --experiment bottleneck
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import json
from pathlib import Path
from typing import Dict, List, Optional
import time


# 数据集配置
DATASET_CONFIGS = {
    "juelich": {
        "name": "Jülich Pedestrian Dynamics Data Archive",
        "description": "最权威的行人动力学实验数据库，专门用于疏散研究",
        "website": "https://ped.fz-juelich.de/da/",
        "doi": "10.34735/ped.da",
        "license": "CC BY 4.0",
        "base_url": "http://ped.fz-juelich.de/experiments",
        "experiments": {
            "bottleneck": {
                "name": "Bottleneck Experiments",
                "doi": "10.34735/ped.2009.6",
                "description": "瓶颈通过实验，不同宽度",
                "page_id": "hermes_bottleneck",
                "page_url": "https://ped.fz-juelich.de/da/doku.php?id=hermes_bottleneck",
                "data_dir": "2009.05.12_Duesseldorf_Messe_Hermes/data/zip",
                "trajectories_txt": "2009bottleneck_trajectories_txt.zip",
                "trajectories_hdf5": "2009bottleneck_trajectories_hdf5.zip",
                "metadata": "2009bottleneck_metadata.json",
            },
            "t_junction": {
                "name": "T-Junction",
                "doi": "10.34735/ped.2009.7",
                "description": "T型路口人流",
                "page_id": "tjunction",
                "page_url": "https://ped.fz-juelich.de/da/doku.php?id=tjunction",
                "data_dir": "2009.05.12_Duesseldorf_Messe_Hermes/data/zip",
                "trajectories_txt": "2009tjunction_trajectories_txt.zip",
                "trajectories_hdf5": "2009tjunction_trajectories_hdf5.zip",
                "metadata": "2009tjunction_metadata.json",
            },
            "bottleneck_crowd": {
                "name": "Crowds in front of bottlenecks",
                "doi": "10.34735/ped.2018.1",
                "description": "瓶颈前人群行为（物理+心理）",
                "page_id": "crowdqueue",
                "page_url": "https://ped.fz-juelich.de/da/doku.php?id=crowdqueue",
                "note": "需要从实验页面手动下载或联系获取",
                "contact": "ped-data-archive@fz-juelich.de",
            },
            "bidirectional_corridor": {
                "name": "Bidirectional Corridor Flow",
                "doi": "10.34735/ped.2013.5",
                "description": "双向走廊人流",
                "page_id": "corridor6",
                "page_url": "https://ped.fz-juelich.de/da/doku.php?id=corridor6",
                "note": "需要从实验页面手动下载或联系获取",
                "contact": "ped-data-archive@fz-juelich.de",
            },
            "stairs": {
                "name": "Stairs Evacuation",
                "doi": "10.34735/ped.2009.4",
                "description": "楼梯疏散（体育场）",
                "page_id": "stairsupper",
                "page_url": "https://ped.fz-juelich.de/da/doku.php?id=stairsupper",
                "note": "需要从实验页面手动下载或联系获取",
                "contact": "ped-data-archive@fz-juelich.de",
            },
            "croma": {
                "name": "CroMa Project - Crowd Management",
                "doi": "10.34735/ped.2021.2",
                "description": "交通基础设施人群管理（~1000人）",
                "page_id": "croma",
                "page_url": "https://ped.fz-juelich.de/da/doku.php?id=croma",
                "note": "需要联系获取",
                "contact": "ped-data-archive@fz-juelich.de",
            },
        },
        "note": "部分实验数据需要联系 ped-data-archive@fz-juelich.de 获取",
    },
    "lyon": {
        "name": "Lyon Dense Crowd Dataset",
        "description": "高密度人群数据集（最高4人/m²），约7000条轨迹",
        "website": "https://zenodo.org/records/13830435",
        "doi": "10.5281/zenodo.13830435",
        "license": "CC BY 4.0",
        "zenodo_record": "13830435",
        "year": "2024-2025",
        "publication": "Nature Scientific Data (2025)",
    },
    "grand_central": {
        "name": "Grand Central Station Dataset",
        "description": "火车站大厅场景，12,684条行人轨迹",
        "website": "https://openaccess.thecvf.com/content_cvpr_2015/html/Yi_Understanding_Pedestrian_Behaviors_2015_CVPR_paper.html",
        "citation": "Yi et al., CVPR 2015",
        "note": "需要联系作者获取数据",
        "contact": "联系论文作者: Shuai Yi, Hongsheng Li, Xiaogang Wang",
    },
    "eth_ucy": {
        "name": "ETH-UCY Pedestrian Trajectory Dataset",
        "description": "经典行人轨迹基准数据集",
        "website": "https://github.com/vita-epfl/trajnetplusplusdata",
        "base_url": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/master/train/real_data",
        "datasets": {
            "biwi_eth": "biwi_eth.ndjson",
            "biwi_hotel": "biwi_hotel.ndjson",
            "crowds_zara01": "crowds_zara01.ndjson",
            "crowds_zara02": "crowds_zara02.ndjson",
        },
    },
}


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """下载文件"""
    try:
        print(f"  [下载] {description or output_path.name}...")
        print(f"    来源: {url}")
        
        # 创建进度显示
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\r    进度: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, show_progress)
        print()  # 换行
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  [完成] 保存到: {output_path}")
        print(f"  [大小] {file_size:.2f} MB")
        return True
    except Exception as e:
        print(f"  [错误] 下载失败: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_from_zenodo(record_id: str, output_dir: Path) -> bool:
    """从 Zenodo 下载数据集"""
    try:
        # Zenodo API 获取文件列表
        api_url = f"https://zenodo.org/api/records/{record_id}"
        print(f"  [查询] Zenodo 记录: {record_id}")
        
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read())
        
        files = data.get("files", [])
        if not files:
            print("  [警告] 未找到可下载文件")
            return False
        
        print(f"  [找到] {len(files)} 个文件")
        
        success_count = 0
        for file_info in files:
            file_url = file_info["links"]["self"]
            filename = file_info["key"]
            output_path = output_dir / filename
            
            if output_path.exists():
                print(f"  [跳过] {filename} 已存在")
                success_count += 1
                continue
            
            if download_file(file_url, output_path, filename):
                success_count += 1
        
        return success_count > 0
    except Exception as e:
        print(f"  [错误] Zenodo 下载失败: {e}")
        return False


def download_juelich_experiment(experiment_name: str, output_dir: Path) -> bool:
    """下载 Jülich 实验数据"""
    config = DATASET_CONFIGS["juelich"]
    experiments = config["experiments"]
    base_url = config["base_url"]
    
    if experiment_name not in experiments:
        print(f"  [错误] 未知实验: {experiment_name}")
        print(f"  可用实验: {', '.join(experiments.keys())}")
        return False
    
    exp_config = experiments[experiment_name]
    
    # 检查是否有直接下载链接
    if "data_dir" not in exp_config:
        print(f"  [提示] {exp_config['name']} 需要手动下载")
        if "page_url" in exp_config:
            print(f"  实验页面: {exp_config['page_url']}")
        if "note" in exp_config:
            print(f"  说明: {exp_config['note']}")
        print(f"  联系邮箱: {exp_config.get('contact', 'ped-data-archive@fz-juelich.de')}")
        print(f"  DOI: {exp_config.get('doi', 'N/A')}")
        return False
    
    # 创建实验目录
    exp_dir = output_dir / "juelich" / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # 下载轨迹数据（txt格式，推荐）
    if "trajectories_txt" in exp_config:
        data_dir = exp_config["data_dir"]
        filename = exp_config["trajectories_txt"]
        url = f"{base_url}/{data_dir}/{filename}"
        zip_path = exp_dir / filename
        
        if download_file(url, zip_path, f"{exp_config['name']} - 轨迹数据 (TXT)"):
            # 解压
            try:
                print(f"  [解压] {zip_path.name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(exp_dir)
                zip_path.unlink()  # 删除zip文件
                print(f"  [完成] 轨迹数据已解压")
                success_count += 1
            except Exception as e:
                print(f"  [错误] 解压失败: {e}")
    
    # 下载元数据
    if "metadata" in exp_config:
        data_dir = exp_config["data_dir"]
        filename = exp_config["metadata"]
        url = f"{base_url}/{data_dir}/{filename}"
        metadata_path = exp_dir / filename
        
        if download_file(url, metadata_path, f"{exp_config['name']} - 元数据"):
            success_count += 1
    
    # 保存实验信息
    if success_count > 0:
        experiment_info = {
            "experiment": experiment_name,
            "name": exp_config["name"],
            "doi": exp_config.get("doi"),
            "description": exp_config.get("description"),
            "page_url": exp_config.get("page_url"),
            "download_date": time.strftime("%Y-%m-%d"),
            "files_downloaded": success_count,
        }
        with open(exp_dir / "experiment_info.json", 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, indent=2, ensure_ascii=False)
        
        print(f"  [完成] 数据已保存到: {exp_dir}")
        return True
    
    return False


def download_eth_ucy(output_dir: Path) -> bool:
    """下载 ETH-UCY 数据集"""
    config = DATASET_CONFIGS["eth_ucy"]
    base_url = config["base_url"]
    datasets = config["datasets"]
    
    data_dir = output_dir / "eth_ucy"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n下载 ETH-UCY 数据集")
    print(f"  来源: {config['website']}")
    
    success_count = 0
    for name, filename in datasets.items():
        url = f"{base_url}/{filename}"
        output_path = data_dir / filename
        
        if output_path.exists():
            print(f"  [跳过] {name} 已存在")
            success_count += 1
            continue
        
        if download_file(url, output_path, name):
            success_count += 1
    
    if success_count > 0:
        # 保存元数据
        metadata = {
            "name": config["name"],
            "website": config["website"],
            "download_date": time.strftime("%Y-%m-%d"),
            "files": list(datasets.values()),
        }
        with open(data_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return success_count > 0


def download_lyon_dataset(output_dir: Path) -> bool:
    """下载 Lyon 高密度人群数据集"""
    config = DATASET_CONFIGS["lyon"]
    record_id = config["zenodo_record"]
    
    data_dir = output_dir / "lyon_dense_crowd"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n下载 Lyon Dense Crowd Dataset")
    print(f"  描述: {config['description']}")
    print(f"  发表: {config['publication']}")
    print(f"  DOI: {config['doi']}")
    print(f"  年份: {config['year']}")
    
    return download_from_zenodo(record_id, data_dir)


def list_available_datasets():
    """列出所有可用数据集"""
    print("=" * 70)
    print("可用的人群疏散数据集")
    print("=" * 70)
    
    for key, config in DATASET_CONFIGS.items():
        print(f"\n[{key.upper()}] {config['name']}")
        print(f"  描述: {config['description']}")
        
        if "doi" in config:
            print(f"  DOI: {config['doi']}")
        if "website" in config:
            print(f"  网址: {config['website']}")
        if "license" in config:
            print(f"  许可: {config['license']}")
        
        if key == "juelich" and "experiments" in config:
            print(f"\n  可用实验:")
            for exp_key, exp_config in config["experiments"].items():
                status = "✓" if exp_config.get("url") else "⚠ (需联系)"
                print(f"    {status} {exp_key}: {exp_config['name']}")
                if exp_config.get("doi"):
                    print(f"       DOI: {exp_config['doi']}")


def main():
    parser = argparse.ArgumentParser(
        description="下载人群疏散相关的公开数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有可用数据集
  python scripts/download_evacuation_datasets.py --list

  # 下载 Jülich 瓶颈实验数据
  python scripts/download_evacuation_datasets.py --dataset juelich --experiment bottleneck

  # 下载所有 Jülich 实验
  python scripts/download_evacuation_datasets.py --dataset juelich --experiment all

  # 下载 Lyon 数据集
  python scripts/download_evacuation_datasets.py --dataset lyon

  # 下载 ETH-UCY 数据集
  python scripts/download_evacuation_datasets.py --dataset eth_ucy

  # 下载所有数据集
  python scripts/download_evacuation_datasets.py --dataset all
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["juelich", "lyon", "grand_central", "eth_ucy", "all"],
        help="要下载的数据集"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Jülich 实验名称（如 bottleneck, t_junction），或 'all' 下载所有"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="数据保存目录（默认: data/raw）"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用数据集"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
        return
    
    if not args.dataset:
        print("错误: 请指定 --dataset 或使用 --list 查看可用数据集")
        parser.print_help()
        return
    
    # 创建输出目录
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("人群疏散数据集下载工具")
    print("=" * 70)
    print(f"\n数据保存目录: {output_dir}")
    
    success_count = 0
    
    # 下载数据集
    if args.dataset == "all" or args.dataset == "juelich":
        print("\n" + "=" * 70)
        print("Jülich Pedestrian Dynamics Data Archive")
        print("=" * 70)
        config = DATASET_CONFIGS["juelich"]
        print(f"名称: {config['name']}")
        print(f"描述: {config['description']}")
        print(f"网站: {config['website']}")
        print(f"DOI: {config['doi']}")
        print(f"许可: {config['license']}")
        
        if args.experiment:
            if args.experiment == "all":
                # 下载所有可用实验
                for exp_name in config["experiments"].keys():
                    if download_juelich_experiment(exp_name, output_dir):
                        success_count += 1
            else:
                if download_juelich_experiment(args.experiment, output_dir):
                    success_count += 1
        else:
            print("\n提示: 使用 --experiment <name> 指定实验，或 --experiment all 下载所有")
            print("可用实验:")
            for exp_key, exp_config in config["experiments"].items():
                status = "✓" if exp_config.get("url") else "⚠ (需联系)"
                print(f"  {status} {exp_key}: {exp_config['name']}")
    
    if args.dataset == "all" or args.dataset == "lyon":
        print("\n" + "=" * 70)
        if download_lyon_dataset(output_dir):
            success_count += 1
    
    if args.dataset == "all" or args.dataset == "eth_ucy":
        print("\n" + "=" * 70)
        if download_eth_ucy(output_dir):
            success_count += 1
    
    if args.dataset == "grand_central":
        print("\n" + "=" * 70)
        print("Grand Central Station Dataset")
        print("=" * 70)
        config = DATASET_CONFIGS["grand_central"]
        print(f"名称: {config['name']}")
        print(f"描述: {config['description']}")
        print(f"提示: {config['note']}")
        print(f"联系: {config['contact']}")
        print(f"网站: {config['website']}")
    
    # 总结
    print("\n" + "=" * 70)
    print("下载完成")
    print("=" * 70)
    print(f"\n数据目录: {output_dir}")
    print(f"成功下载: {success_count} 个数据集/实验")
    print("\n重要提示:")
    print("1. 使用数据时请引用相应的 DOI")
    print("2. 遵守数据许可协议（通常为 CC BY 4.0）")
    print("3. 部分数据需要从实验页面手动下载")
    print("4. 访问 https://ped.fz-juelich.de/da/ 查看所有可用实验")
    print("\n数据格式说明:")
    print("- 轨迹数据格式: ID, frame, x-coordinate [cm], y-coordinate [cm], z-coordinate [cm]")
    print("- 坐标单位: 厘米 (cm)，使用时需要转换为米 (m)")
    print("- 帧率: 通常为 16 fps 或 25 fps，请查看元数据确认")


if __name__ == "__main__":
    main()
