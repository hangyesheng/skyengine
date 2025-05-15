class Job:
    def __init__(self, job_id, target_count=None):
        self.id = job_id
        self.operations = []
        self.target_count = target_count  # 目标处理工件数，可选

    def add_operation(self, op):
        self.operations.append(op)

    def get_target_count(self):
        return self.target_count

    def update_target_count(self):
        """
        在最后一个 Operation 完成处理后调用，用于统计任务完成度。
        """
        self.target_count+=1
