from xml.sax import parse

import yaml


def parse_brandimarte_data(filepath):
    """
    解析基于图拓扑的 AGV 实例文件
    
    新格式说明：
    - 第一行：<作业数量> <机器数量> <AGV数量> <节点数量> <边数量>
    - 接下来 N 行：作业工序数据
    - 接下来 P 行：节点信息 (point_id, x, y)
    - 接下来 L 行：边信息 (link_id, point1_id, point2_id, weight)
    - 接下来 M 行：机器绑定 (machine_id, point_id)
    - 最后 A 行：AGV配置 (agv_id, point_id, velocity)
    
    Args:
        filepath: AGV 实例文件路径
        
    Returns:
        dict: 包含 jobs, points, links, machines, agvs 的字典
    """
    with open(filepath, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    idx = 0
    
    # 解析第一行：作业数量、机器数量、AGV数量、节点数量、边数量
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
        "machines": machines,  # [(machine_id, point_id), ...]
        "agvs": agvs  # [(agv_id, point_id, velocity), ...]
    }


def generate_map_config(parsed_data):
    """
    生成基于图拓扑的 map_config
    
    Args:
        parsed_data: 解析后的数据，包含 points, links, machines, agvs
        
    Returns:
        dict: map_config 字典
    """
    # 构建 Points
    points = [
        {
            "point": {
                "id": point_id,
                "coordinate": [x, y]
            }
        }
        for point_id, x, y in parsed_data['points']
    ]
    
    # 构建 Links（带权重）
    links = [
        {
            "link": {
                "id": link_id,
                "begin": point1_id,
                "end": point2_id,
                "weight": weight
            }
        }
        for link_id, point1_id, point2_id, weight in parsed_data['links']
    ]
    
    # 构建 Machines（通过 point_id 关联）
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
    
    # 构建 AGVs（通过 point_id 关联）
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

    map_config = {
        "config": {
            "width": 20,
            "height": 30,
            "block_count": 0,
            "point_count": len(points),
            "machine_count": len(machines),
            "path_count": len(links),
            "agv_count": len(agvs),
            "blocks": [],
            "points": points,
            "machines": machines,
            "links": links,  # 注意：使用 links 而非 paths
            "agvs": agvs
        }
    }
    return map_config

def generate_job_config(parsed_data):
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
                "id": op_idx,
                "candidates": [
                    {"machine": m, "duration": d}
                    for m, d in machine_options
                ]
            }
            job_entry["job"]["operations"].append(op_entry)
        jobs_yaml.append(job_entry)

    job_config = {
        "job_config": {
            "job_count": parsed_data["job_count"],
            "machine_count": parsed_data["machine_count"],
            "jobs": jobs_yaml
        }
    }

    return job_config
