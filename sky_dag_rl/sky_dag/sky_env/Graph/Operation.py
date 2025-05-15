class Operation:
    def __init__(self, op_id, cpu_req, mem_req,duration=100):
        # operation本身的属性
        self.id = op_id
        self.cpu_req = cpu_req
        self.mem_req = mem_req

        # operation本身的状态
        self.state = "blocked"
        self.progress = 0.0
        self.dependencies = []
        self.successors = []
        self.assigned_node = None
        self.duration = duration # 当前产品需要加工的进度条

        # operation处理item相关的状态
        self.processed_item_list = []
        self.current_progress = 0  # 当前物品的加工进度
        self.current_start_time = None

        self.status = None

    def add_dependency(self, op):
        """
        添加依赖节点
        :param op:
        :return:
        """
        self.dependencies.append(op)

    def add_successor(self, op):
        """
        添加依赖节点
        :param op:
        :return:
        """
        self.successors.append(op)

    def is_ready(self):
        return all(dep.state == "finished" for dep in self.dependencies)

    # ----------正常运行阶段----------
    def get_feature(self):
        return {
            "id": self.id,
            "state": self.state,
            "cpu": self.cpu_req,
            "mem": self.mem_req,
            "processed_item_list": self.processed_item_list,
            "assigned_node": self.assigned_node.id if self.assigned_node else None
        }

    def save_status(self):
        self.status = self.state

    def load_status(self):
        self.state = self.status

    def pause(self):
        """
        暂停operation运行
        :return:
        """
        self.state = "paused"
        self.save_status()

    def resume(self):
        """
        重启运行
        :return:
        """
        self.state = self.load_status()

    def fail(self):
        """
        operation报错
        :return:
        """
        self.state = "fail"

    def step(self, node_speed, env_time, packet=None, error_chance=0.0):
        """
        Operation 的执行周期逻辑。
        每次 step 表示推进一个时间片。

        :param node_speed: 当前节点加工速度 (单位进度/时间)
        :param env_time: 当前环境时间
        :param packet: 传入的工件（可为空）
        :param error_chance: 故障概率（用于模拟）
        :return: 是否完成一个工件（True/False/None）
        """

        # 故障状态不执行任何操作
        if self.state == "failed":
            return None

        # 暂停状态等待外部 resume
        if self.state == "paused":
            return None

        # 尚未满足依赖条件
        if self.state == "blocked":
            if self.is_ready():
                self.state = "ready"
            else:
                return None

        # 准备状态，等待输入工件
        if self.state == "ready":
            if packet:
                self.current_progress = 0.0
                self.current_start_time = env_time
                self.state = "active"
            else:
                return None  # 无输入继续等待

        # 主加工阶段
        if self.state == "active":
            self.current_progress += node_speed

            # 加工完成
            if self.current_progress >= self.duration:
                self.processed_item_list.append({
                    "start_time": self.current_start_time,
                    "end_time": env_time,
                })
                self.current_progress = 0.0
                self.state = "ready"
                return True

        # 空闲状态，等待下一轮调度
        if self.state == "ready":
            return False


    def check_dependencies(self):
        """
        判断前置条件是否满足
        :return:
        """
        if self.state == "pending" and self.is_ready():
            self.state = "ready"

    def cal_qos(self, current_time):
        """
        查看运行质量
        :return:
        """
        if hasattr(self, "deadline"):
            return 1 if current_time <= self.deadline else 0
        return 1  # 默认不罚分

    # ----------重调度阶段----------
    def request(self):
        """
        请求资源
        :return:
        """
        return {"cpu": self.cpu_req, "mem": self.mem_req}

    def choose(self, candidate_nodes):
        """
        进行物理机的选择
        :return:
        """
        # 默认选择资源最多的节点，或在外部策略中完成
        sorted_nodes = sorted(candidate_nodes, key=lambda n: n.cpu_capacity + n.mem_capacity, reverse=True)
        return sorted_nodes[0] if sorted_nodes else None

    def assign_to_node(self, node, time):
        """
        分配到节点上
        :param node:
        :return:
        """
        self.assigned_node = node
        self.state = "paused"
