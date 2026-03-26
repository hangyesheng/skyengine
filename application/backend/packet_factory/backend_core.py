import os
import threading
from typing import List
from typing import Callable
import json
import time

import yaml

from executor.packet_factory.lifecycle.bootstrap import bootstrap
from executor.packet_factory.packet_factory.packet_factory_env.packet_factory_env import PacketFactoryEnv
from executor.packet_factory.packet_factory.packet_factory_env.Job.Job import Job
from executor.packet_factory.packet_factory.packet_factory_env.Machine.Machine import Machine
from executor.packet_factory.packet_factory.packet_factory_env.Agv.AGV import AGV
import config

from application.backend.packet_factory.service import file_service

from executor.packet_factory.logger.logger import Logger
from executor.packet_factory.packet_factory.Agent.BaseAgent import TRAINING, EVALUATION, INFERENCE

LOGGER = Logger(log_path=config.BACKEND_LOG_DIR, name="backend").logger


class ThreadPool:
    def __init__(self):
        self.threads = []
        self.stop_event = threading.Event()

    def __len__(self):
        return len(self.threads)

    def submit(self, func: Callable, *args, **kwargs):
        """提交任务到线程池,确保当前只有单个在修改env"""
        if len(self.threads) >= 1:
            self.shutdown()
        try:
            thread = threading.Thread(target=func, args=(self.stop_event, *args), kwargs=kwargs)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        except Exception as e:
            LOGGER.error(e)

    def shutdown(self, wait=True):
        """请求线程退出（通过事件标志）"""
        self.stop_event.set()  # 通知线程退出
        if wait:
            for thread in self.threads:
                thread.join()
        self.threads = []  # 清空缓冲池
        self.stop_event.clear()  # 清除退出标志
        LOGGER.info("[INFO] 线程池已关闭")


class BackendCore:
    def __init__(self):
        # 环境本身
        self.env: PacketFactoryEnv = None
        # 线程池，但是实际按照当前的 core 实现大概只能有单个线程 (受到 env 限制)...后续再修改吧!
        self.thread_pool = ThreadPool()
        # 内存中的配置存储 {config_name: config_data}
        self.config_store = {}

    def bootstrap(self, stop_event: threading.Event, target_factory: str):
        specific_config = self.config_store[target_factory]
        LOGGER.info(f"[Bootstrap] 从内存加载配置：{target_factory}")

        template_config_path = os.path.join(file_service.get_config_dir(), 'application_config.yaml')

        with open(template_config_path, 'r') as f:
            template_config = yaml.safe_load(f)

        final_config = template_config['config']
        env_type = final_config['env_type']
        final_config[env_type]['job_config'] = specific_config['job_config']
        final_config[env_type]['event_config'] = specific_config['event_config']
        final_config[env_type]['map_config'] = specific_config['map_config']

        # 创建环境与智能体
        env, agent = bootstrap(final_config)

        self.env = env

        # 获取 Agent 模式
        agent_mode = getattr(agent, 'mode', TRAINING)
        LOGGER.info(f"[Bootstrap] Agent mode: {agent_mode}, model_path: {getattr(agent, 'model_path', None)}")

        # 重置环境
        observations = env.reset()

        # 根据模式执行不同的流程
        if agent_mode == TRAINING:
            self._run_training(env, agent, stop_event)
        elif agent_mode == EVALUATION:
            self._run_evaluation(env, agent, stop_event)
        elif agent_mode == INFERENCE:
            self._run_inference(env, agent, stop_event)
        else:
            LOGGER.warning(f"[Bootstrap] Unknown agent mode: {agent_mode}, running default loop")
            self._run_default(env, agent, stop_event)

    def _run_training(self, env, agent, stop_event: threading.Event):
        """
        训练模式：快速迭代，不渲染
        :param env: 环境
        :param agent: Agent
        :param stop_event: 停止事件
        """
        LOGGER.info("[Training] Starting training mode...")
        
        # 运行一个 episode（直到结束）
        while not (env.env_is_finished() and not stop_event.is_set()):
            # 输入获得环境状态并决策
            actions = env.action_space(agent)

            # agent 在外部决策
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # 训练更新
            if hasattr(agent, 'update'):
                agent.update(observations, rewards)

        
        # 保存训练结果
        self._save_training_results(env, agent)
        
        LOGGER.info(f"[Training] Total makespan: {env.env_timeline}s")

    def _run_evaluation(self, env, agent, stop_event: threading.Event):
        """
        评估模式：快速迭代，生成评估报告
        :param env: 环境
        :param agent: Agent
        :param stop_event: 停止事件
        """
        LOGGER.info("[Evaluation] Starting evaluation mode...")
        
        # 验证 model_path
        model_path = getattr(agent, 'model_path', None)
        if not model_path:
            LOGGER.error("[Evaluation] model_path is required for evaluation mode")
            return
        
        # 运行一个 episode（直到结束）
        while not (env.env_is_finished() and not stop_event.is_set()):
            # 输入获得环境状态并决策
            actions = env.action_space(agent)

            # agent 在外部决策
            observations, rewards, terminations, truncations, infos = env.step(actions)

        
        # 生成评估报告
        self._generate_evaluation_report(env, agent)
        
        LOGGER.info(f"[Evaluation] Total makespan: {env.env_timeline}s")

    def _run_inference(self, env, agent, stop_event: threading.Event):
        """
        推理模式：启用可视化，实时推送状态到前端
        :param env: 环境
        :param agent: Agent
        :param stop_event: 停止事件
        """
        LOGGER.info("[Inference] Starting inference mode with visualization...")
        
        # 验证 model_path
        model_path = getattr(agent, 'model_path', None)
        if not model_path:
            LOGGER.error("[Inference] model_path is required for inference mode")
            return
        
        # 运行一个 episode（直到结束）
        while not (env.env_is_finished() and not stop_event.is_set()):
            # 输入获得环境状态并决策
            actions = env.action_space(agent)

            # agent 在外部决策
            observations, rewards, terminations, truncations, infos = env.step(actions)

        
        LOGGER.info(f"[Inference] Total makespan: {env.env_timeline}s")

    def _run_default(self, env, agent, stop_event: threading.Event):
        """
        默认模式：兼容旧版本
        :param env: 环境
        :param agent: Agent
        :param stop_event: 停止事件
        """
        # 运行一个 episode（直到结束）
        while not (env.env_is_finished() and not stop_event.is_set()):
            # 输入获得环境状态并决策
            actions = env.action_space(agent)

            # agent 在外部决策
            observations, rewards, terminations, truncations, infos = env.step(actions)

        LOGGER.info(f"total makespan: {env.env_timeline}s")

    def _save_training_results(self, env, agent):
        """
        保存训练结果到规范目录
        :param env: 环境实例
        :param agent: Agent 实例
        """
        try:
            # 优先调用 Agent 的 save_training_result 方法（如果存在）
            if hasattr(agent, 'save_training_result'):
                result_path = agent.save_training_result()
                if result_path:
                    LOGGER.info(f"[Training] Results saved to {result_path}")
            else:
                # 降级方案：保存到 training_logs/results/ 目录
                agent_name = getattr(agent, 'name', 'UnknownAgent')
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                result_dir = f"training_logs/results/{agent_name}_{timestamp}"
                
                os.makedirs(result_dir, exist_ok=True)
                
                results = {
                    'makespan': env.env_timeline,
                    'decision_stats': agent.get_decision_stats() if hasattr(agent, 'get_decision_stats') else {},
                    'q_table_size': len(getattr(agent, 'q_table', {})),
                    'metadata': {
                        'agent_name': agent_name,
                        'agent_id': getattr(agent, 'agent_id', None),
                        'save_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
                
                result_file = os.path.join(result_dir, 'training_report.json')
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                LOGGER.info(f"[Training] Results saved to {result_file}")
            
            # 保存模型（使用 Agent 的默认路径或降级方案）
            if hasattr(agent, 'save_model'):
                model_path = agent.save_model()
                if model_path:
                    LOGGER.info(f"[Training] Model saved to {model_path}")
            else:
                # 降级方案：保存到 training_logs/models/ 目录
                agent_name = getattr(agent, 'name', 'UnknownAgent')
                model_dir = f"training_logs/models/{agent_name}"
                os.makedirs(model_dir, exist_ok=True)
                
                model_file = os.path.join(model_dir, 'agent_model.json')
                model_data = {
                    'metadata': {
                        'agent_name': agent_name,
                        'agent_id': getattr(agent, 'agent_id', None),
                        'save_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
                
                with open(model_file, 'w') as f:
                    json.dump(model_data, f, indent=2)
                
                LOGGER.info(f"[Training] Model saved to {model_file}")
                
        except Exception as e:
            LOGGER.error(f"[Training] Failed to save results: {e}")

    def _generate_evaluation_report(self, env, agent):
        """
        生成评估报告并保存到规范目录
        :param env: 环境实例
        :param agent: Agent 实例
        """
        try:
            # 准备评估结果
            report = {
                'mode': EVALUATION,
                'model_path': getattr(agent, 'model_path', None),
                'makespan': env.env_timeline,
                'decision_stats': agent.get_decision_stats() if hasattr(agent, 'get_decision_stats') else {},
                'q_table_size': len(getattr(agent, 'q_table', {})),
                'epsilon': getattr(agent, 'epsilon', 0),
                'metadata': {
                    'agent_name': getattr(agent, 'name', 'UnknownAgent'),
                    'agent_id': getattr(agent, 'agent_id', None),
                    'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            # 保存评估报告
            agent_name = getattr(agent, 'name', 'UnknownAgent')
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            eval_dir = f"training_logs/evaluations/{agent_name}_{timestamp}"
            os.makedirs(eval_dir, exist_ok=True)

            result_file = os.path.join(eval_dir, 'evaluation_report.json')
            with open(result_file, 'w') as f:
                json.dump(report, f, indent=2)

            LOGGER.info(f"[Evaluation] Report saved to {result_file}")

        except Exception as e:
            LOGGER.error(f"[Evaluation] Failed to generate report: {e}")

    def is_factory_alive(self):
        return self.env is not None

    def factory_start(self):
        self.env.env_visualizer.run()

    def factory_pause(self):
        self.env.env_visualizer.pause()

    def factory_reset(self):
        self.env.env_visualizer.restart()

    def change_factory_speed(self, speed_level: int):
        self.env.env_visualizer.change_speed(speed_level)

    def get_agvs(self):
        if self.env is None:
            return []
        agvs: List[AGV] = self.env.getAGVs()
        agv_list = [{"id": agv.id} for agv in agvs]
        return agv_list

    def pause_agv(self, agv_id):
        self.env.env_visualizer.pause_agv(agv_id)

    def resume_agv(self, agv_id):
        self.env.env_visualizer.resume_agv(agv_id)

    def get_machines(self):
        if self.env is None:
            return []
        machines: List[Machine] = self.env.getMachines()
        machine_list = [{"id": machine.id} for machine in machines]
        return machine_list

    def pause_machine(self, machine_id):
        self.env.env_visualizer.pause_machine(machine_id)

    def resume_machine(self, machine_id):
        self.env.env_visualizer.resume_machine(machine_id)

    def get_job_templates(self):
        if self.env is None:
            return []
        jobs: List[Job] = self.env.getJobTemplates()
        job_list = [{"id": job.id} for job in jobs]
        return job_list

    def add_job(self, job_id: int):
        self.env.env_visualizer.add_job(job_id)

    def get_jobs_progress(self):
        if self.env is None:
            return []
        jobs: List[Job] = self.env.getJobs()
        job_list = [{"id": job.id, "status": job.get_status().name, "progress": round(job.get_progress() * 100.0, 2)}
                    for job in jobs]
        return job_list

    def render_map(self, target_factory):
        """插入配置文件，启动当前渲染地图"""
        # 启动系统!
        self.thread_pool.submit(self.bootstrap, target_factory)

    def save_config_to_memory(self, config_name: str, config_data: dict):
        """将配置保存到内存"""
        self.config_store[config_name] = config_data
        LOGGER.info(f"[Config] 配置已保存到内存：{config_name}")

    def get_config_from_memory(self, config_name: str):
        """从内存获取配置"""
        return self.config_store.get(config_name)

    def get_all_config_names(self):
        """获取所有配置名称列表"""
        return list(self.config_store.keys())

    def get_map_current(self):
        pic = b''
        try:
            pic = self.env.env_visualizer.get_map()
        except Exception as e:
            print(e)
        return pic
