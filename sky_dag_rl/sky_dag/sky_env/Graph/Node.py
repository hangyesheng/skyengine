class Node:
    def __init__(self, node_id, position, cpu_capacity, mem_capacity, execution_factor=1.0):
        self.id = node_id
        self.position = position
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity
        self.running_operations = []
        self.connections = []  # Underlay: 连接到的其他 Node，带传输延迟
        self.is_failed = False
        self.execution_factor = execution_factor

    def connect_to(self, other_node, delay=1):
        self.connections.append({"target": other_node, "delay": delay})

    def can_run(self, op):
        if self.is_failed:
            return False
        used_cpu = sum(o.cpu_req for o in self.running_operations)
        used_mem = sum(o.mem_req for o in self.running_operations)
        return (used_cpu + op.cpu_req <= self.cpu_capacity and
                used_mem + op.mem_req <= self.mem_capacity)

    def assign_operation(self, op):
        if self.can_run(op):
            op.assign_to_node(self)
            self.running_operations.append(op)
            return True
        return False

    # ----------正常运行阶段----------
    def monitor(self):
        """
        获取节点信息
        :return:
        """
        return {
            "id": self.id,
            "resource": self.get_resource_vector(),
            "ops": [op.id for op in self.running_operations],
            "status": "failed" if self.is_failed else "running"
        }

    def get_resource_vector(self):
        """
        获取资源向量
        :return:
        """
        used_cpu = sum(op.cpu_req for op in self.running_operations)
        used_mem = sum(op.mem_req for op in self.running_operations)
        return {
            "cpu_used": used_cpu,
            "cpu_total": self.cpu_capacity,
            "mem_used": used_mem,
            "mem_total": self.mem_capacity,
            "running_op_count": len(self.running_operations)
        }

    def step(self, env_time):
        """
        节点执行一个时间周期的逻辑，推进所有正在运行的操作。

        :param env_time: 当前全局环境时间
        :return: list of (op_id, processed_item)
        """
        finished_items = []

        for op in self.running_operations:
            # 每个 Operation 都有自己的输入队列，按需传入数据
            input_queue = getattr(op, "input_queue", [])
            packet = input_queue.pop(0) if input_queue else None

            result = op.step(
                node_speed=self.execution_factor,
                env_time=env_time,
                packet=packet
            )

            # 若成功处理完成了一个工件，记录信息
            if result is True:
                finished_items.append((op.id, op.current_packet))

        return finished_items

    # ----------重调度阶段----------
    def available(self):
        """
        给出当前的可用资源
        :return:
        """
        used_cpu = sum(op.cpu_req for op in self.running_operations)
        used_mem = sum(op.mem_req for op in self.running_operations)
        return {
            "cpu_avail": self.cpu_capacity - used_cpu,
            "mem_avail": self.mem_capacity - used_mem
        }

    def preempt(self, op=None):
        """
        抢占当前的物理机
        :return:
        """
        if op:
            if op in self.running_operations:
                self.running_operations.remove(op)
                op.state = "waiting"
        else:
            for op in self.running_operations:
                op.state = "waiting"
            self.running_operations.clear()

    def fail(self):
        self.is_failed = True

    def recover(self):
        self.is_failed = False
