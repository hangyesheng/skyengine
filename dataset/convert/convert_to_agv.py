"""
FJSP 实例转 AGV 版本转换脚本（基于带权无向图拓扑）

功能：
1. 读取 dataset/fjsp-instances 下的所有 .txt 文件
2. 为每个实例添加基于带权无向图的工厂拓扑结构
3. 保存到 dataset/agv-instances 目录下，保持相同的目录结构

AGV 版本格式说明：
- 第一行：<作业数量N> <机器数量M> <AGV数量A> <节点数量P> <边数量L>
- 接下来 N 行：作业工序数据（与原格式相同）
- 接下来 P 行：节点/点信息 (point_id, x, y)
- 接下来 L 行：边/链接信息 (link_id, point1_id, point2_id, weight)
- 接下来 M 行：机器绑定信息 (machine_id, point_id)
- 最后 A 行：AGV 配置信息 (agv_id, point_id, velocity)

注意：
- 移除了机器和 AGV 的直接坐标信息
- 使用 Point（节点）+ Link（带权边）构建工厂拓扑图
- Machine 和 AGV 通过 point_id 关联到图中的节点
- Link 的 weight 表示通行代价（时间/距离）
"""

import os
import json
from pathlib import Path


def parse_fjsp_instance(filepath):
    """
    解析 FJSP 实例文件
    
    Args:
        filepath: FJSP 实例文件路径
        
    Returns:
        dict: 包含 job_count, machine_count, jobs 的字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析第一行：作业数量和机器数量
    first_line = lines[0].strip().split()
    job_count = int(first_line[0])
    machine_count = int(first_line[1])
    
    # 解析作业数据
    jobs = []
    for i in range(1, job_count + 1):
        jobs.append(lines[i].strip())
    
    return {
        'job_count': job_count,
        'machine_count': machine_count,
        'jobs': jobs,
        'original_lines': lines
    }


def generate_graph_topology(machine_count, grid_size=20):
    """
    生成基于网格的带权无向图拓扑结构
    
    Args:
        machine_count: 机器数量
        grid_size: 网格大小
        
    Returns:
        tuple: (points, links, machine_point_mapping)
            - points: 节点列表 [(point_id, x, y), ...]
            - links: 边列表 [(link_id, point1_id, point2_id, weight), ...]
            - machine_point_mapping: 机器到节点的映射 {machine_idx: point_id}
    """
    # 计算网格行列数（确保能容纳所有机器）
    cols = int(machine_count ** 0.5) + 1
    rows = (machine_count + cols - 1) // cols
    
    # 生成节点（Point）- 在网格交叉点放置节点
    points = []
    point_id_counter = 1
    point_grid = {}  # (row, col) -> point_id
    
    for row in range(rows + 1):  # +1 为了有额外的连接路径
        for col in range(cols + 1):
            x = col * (grid_size // cols)
            y = row * (grid_size // rows)
            point_id = point_id_counter
            points.append((point_id, x, y))
            point_grid[(row, col)] = point_id
            point_id_counter += 1
    
    # 生成边（Link）- 连接相邻节点形成网格图
    links = []
    link_id_counter = 1
    
    # 水平边
    for row in range(rows + 1):
        for col in range(cols):
            point1_id = point_grid[(row, col)]
            point2_id = point_grid[(row, col + 1)]
            # 权重可以基于距离或随机生成（模拟不同的通行难度）
            weight = 1.0 + (row + col) % 3 * 0.5  # 1.0, 1.5, 2.0 交替
            links.append((link_id_counter, point1_id, point2_id, weight))
            link_id_counter += 1
    
    # 垂直边
    for row in range(rows):
        for col in range(cols + 1):
            point1_id = point_grid[(row, col)]
            point2_id = point_grid[(row + 1, col)]
            weight = 1.0 + (row + col) % 3 * 0.5
            links.append((link_id_counter, point1_id, point2_id, weight))
            link_id_counter += 1
    
    # 为每个机器分配一个节点（优先选择网格内部节点）
    machine_point_mapping = {}
    available_points = list(point_grid.values())
    
    # 简单策略：均匀分配机器到不同节点
    step = max(1, len(available_points) // machine_count)
    for machine_idx in range(machine_count):
        point_idx = (machine_idx * step) % len(available_points)
        machine_point_mapping[machine_idx] = available_points[point_idx]
    
    return points, links, machine_point_mapping


def determine_agv_count(job_count, machine_count):
    """
    根据作业和机器数量确定合适的 AGV 数量
    
    Args:
        job_count: 作业数量
        machine_count: 机器数量
        
    Returns:
        int: AGV 数量
    """
    # 简单的启发式规则：AGV 数量约为机器数量的 1/3 到 1/2
    # 最少 1 个，最多不超过机器数量
    agv_count = max(1, min(machine_count, machine_count // 2))
    return agv_count


def assign_agvs_to_points(points, agv_count):
    """
    为 AGV 分配起始节点
    
    Args:
        points: 节点列表 [(point_id, x, y), ...]
        agv_count: AGV 数量
        
    Returns:
        list: AGV 配置 [(agv_id, point_id, velocity), ...]
    """
    agvs = []
    point_ids = [p[0] for p in points]
    
    for agv_id in range(agv_count):
        # 均匀分布 AGV 到不同节点
        point_idx = agv_id % len(point_ids)
        point_id = point_ids[point_idx]
        velocity = 1.0  # 默认速度
        agvs.append((agv_id + 1, point_id, velocity))
    
    return agvs


def convert_to_agv_format(parsed_data, output_filepath):
    """
    将 FJSP 数据转换为基于图拓扑的 AGV 格式并写入文件
    
    Args:
        parsed_data: 解析后的 FJSP 数据
        output_filepath: 输出文件路径
    """
    job_count = parsed_data['job_count']
    machine_count = parsed_data['machine_count']
    
    # 确定 AGV 数量
    agv_count = determine_agv_count(job_count, machine_count)
    
    # 生成图拓扑结构
    points, links, machine_point_mapping = generate_graph_topology(machine_count)
    
    # 生成 AGV 配置
    agvs = assign_agvs_to_points(points, agv_count)
    
    # 写入文件
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # 第一行：作业数 机器数 AGV数 节点数 边数
        f.write(f"{job_count} {machine_count} {agv_count} {len(points)} {len(links)}\n")
        
        # 作业数据
        for job_line in parsed_data['jobs']:
            f.write(f"{job_line}\n")
        
        # 节点信息 (point_id, x, y)
        for point_id, x, y in points:
            f.write(f"{point_id} {x} {y}\n")
        
        # 边信息 (link_id, point1_id, point2_id, weight)
        for link_id, point1_id, point2_id, weight in links:
            f.write(f"{link_id} {point1_id} {point2_id} {weight}\n")
        
        # 机器绑定信息 (machine_id, point_id) - 机器编号从0开始
        for machine_idx in range(machine_count):
            machine_id = machine_idx  # 从0开始编号
            point_id = machine_point_mapping[machine_idx]
            f.write(f"{machine_id} {point_id}\n")
        
        # AGV 配置信息 (agv_id, point_id, velocity)
        for agv_id, point_id, velocity in agvs:
            f.write(f"{agv_id} {point_id} {velocity}\n")
    
    print(f"  ✓ 已转换: {output_filepath}")
    print(f"    - 作业数: {job_count}, 机器数: {machine_count}, AGV数: {agv_count}")
    print(f"    - 图拓扑: {len(points)} 个节点, {len(links)} 条边")


def convert_directory(source_dir, target_dir):
    """
    转换整个目录下的所有 FJSP 实例（递归处理子目录）
    
    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 递归查找所有 .txt 文件（包括子目录）
    txt_files = list(source_path.rglob('*.txt'))
    
    if not txt_files:
        print(f"⚠ 在 {source_dir} 中未找到 .txt 文件")
        return
    
    print(f"\n处理目录: {source_dir.name}")
    print(f"找到 {len(txt_files)} 个实例文件（包含子目录）")
    
    converted_count = 0
    for txt_file in sorted(txt_files):
        try:
            # 解析原始文件
            parsed_data = parse_fjsp_instance(txt_file)
            
            # 保持相对目录结构
            relative_path = txt_file.relative_to(source_path)
            output_filepath = target_path / relative_path.parent / (txt_file.stem + '_agv.txt')
            
            # 确保输出文件的父目录存在
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换并保存
            convert_to_agv_format(parsed_data, output_filepath)
            converted_count += 1
            
        except Exception as e:
            print(f"  ✗ 转换失败 {txt_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"✓ 成功转换 {converted_count}/{len(txt_files)} 个文件\n")


def main():
    """
    主函数：遍历所有子目录并转换
    """
    # 定义源目录和目标目录
    base_dir = Path(__file__).parent.parent
    source_base = base_dir / 'fjsp-instances'
    target_base = base_dir / 'agv-instances'
    
    print("=" * 70)
    print("FJSP 实例转 AGV 版本转换工具（基于带权无向图拓扑）")
    print("=" * 70)
    print(f"源目录: {source_base}")
    print(f"目标目录: {target_base}")
    print("=" * 70)
    
    # 检查源目录是否存在
    if not source_base.exists():
        print(f"✗ 错误: 源目录不存在: {source_base}")
        return
    
    # 获取所有子目录（递归处理，排除 instances.json 等文件）
    subdirs = [d for d in source_base.iterdir() if d.is_dir()]
    
    if not subdirs:
        print("✗ 错误: 源目录中没有子目录")
        return
    
    print(f"\n找到 {len(subdirs)} 个顶级子目录需要转换:")
    for subdir in subdirs:
        print(f"  - {subdir.name}")
    
    print("\n开始转换...\n")
    
    total_converted = 0
    total_files = 0
    
    # 遍历每个子目录（递归处理所有层级）
    for subdir in sorted(subdirs):
        # 统计文件数量（包括子目录）
        txt_files = list(subdir.rglob('*.txt'))
        total_files += len(txt_files)
        
        # 转换目录（递归处理）
        target_subdir = target_base / subdir.name
        convert_directory(subdir, target_subdir)
        total_converted += len(txt_files)
    
    # 打印总结
    print("=" * 70)
    print("转换完成！")
    print(f"总计处理: {total_files} 个文件")
    print(f"输出目录: {target_base}")
    print("=" * 70)
    
    # 生成转换报告
    report_path = target_base / 'conversion_report.json'
    report = {
        'source_directory': str(source_base),
        'target_directory': str(target_base),
        'total_files_processed': total_files,
        'subdirectories': [d.name for d in subdirs],
        'format_version': 'graph_topology_v2',
        'note': (
            '新版本使用带权无向图表示工厂拓扑结构。\n'
            '文件格式：\n'
            '  1. 第一行: 作业数 机器数 AGV数 节点数 边数\n'
            '  2. 作业数据行\n'
            '  3. 节点信息: point_id x y\n'
            '  4. 边信息: link_id point1_id point2_id weight\n'
            '  5. 机器绑定: machine_id point_id\n'
            '  6. AGV配置: agv_id point_id velocity\n'
            'Machine 和 AGV 不再直接存储坐标，而是通过 point_id 关联到图节点。'
        )
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换报告已保存至: {report_path}")


if __name__ == '__main__':
    main()
