from executor.packet_factory.registry.factory import create_component_by_id

# 这些 key 已在下方显式提取，不应再作为 **kwargs 传入（避免重复参数）
_EXPLICIT_KEYS = frozenset([
    'agent_name', 'name', 'id',
    'ui_mode', 'task_mode', 'model_path',
    'time_limit_seconds', 'fallback_enabled',
])


def initialize_agent(config):
    """
    初始化 Agent，传递 ui_mode、task_mode 和 model_path 参数
    以及 YAML agent 段中的其余参数（device, hidden_dim, gamma 等）
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

    # 将 YAML agent 段中剩余的 key 作为 **kwargs 传入（如 device, hidden_dim, gamma 等）
    extra_kwargs = {k: v for k, v in agent_config.items() if k not in _EXPLICIT_KEYS}

    # 创建 Agent 实例，传递新的维度参数
    agent = create_component_by_id(
        agent_config.get("agent_name"),
        agent_config.get("name"),
        agent_config.get("id"),
        ui_mode=ui_mode,
        task_mode=task_mode,
        model_path=model_path,
        time_limit_seconds=time_limit_seconds,
        fallback_enabled=fallback_enabled,
        **extra_kwargs,
    )

    return agent
