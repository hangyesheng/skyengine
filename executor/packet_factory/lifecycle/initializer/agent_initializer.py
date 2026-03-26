from executor.packet_factory.registry.factory import create_component_by_id

def initialize_agent(config):
    """
    初始化 Agent，传递 mode 和 model_path 参数
    :param config: 配置字典
    :return: Agent 实例
    """
    env_type = config.get("env_type")
    agent_config = config.get(env_type).get("agent")
    
    # 获取强化学习相关配置
    mode = agent_config.get("mode", "training")
    model_path = agent_config.get("model_path")
    
    # 创建 Agent 实例，传递 mode 和 model_path
    agent = create_component_by_id(
        agent_config.get("agent_name"), 
        agent_config.get("name"), 
        agent_config.get("id"),
        mode=mode,
        model_path=model_path
    )
    
    return agent
