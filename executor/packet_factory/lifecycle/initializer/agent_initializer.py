from executor.packet_factory.registry.factory import create_component_by_id

def initialize_agent(config):
    """
    初始化 Agent，传递 ui_mode、task_mode 和 model_path 参数
    :param config: 配置字典
    :return: Agent 实例
    """
    env_type = config.get("env_type")
    agent_config = config.get(env_type).get("agent")

    # 获取强化学习相关配置 - 2x2 模式
    ui_mode = agent_config.get("ui_mode", "backend")      # frontend | backend
    task_mode = agent_config.get("task_mode", "training")  # training | inference
    model_path = agent_config.get("model_path")

    # 获取 OR-Tools 优化器参数
    time_limit_seconds = agent_config.get("time_limit_seconds", 30)
    fallback_enabled = agent_config.get("fallback_enabled", True)

    # 创建 Agent 实例，传递新的维度参数
    agent = create_component_by_id(
        agent_config.get("agent_name"),
        agent_config.get("name"),
        agent_config.get("id"),
        ui_mode=ui_mode,
        task_mode=task_mode,
        model_path=model_path,
        time_limit_seconds=time_limit_seconds,
        fallback_enabled=fallback_enabled
    )

    return agent
