from executor.packet_factory.call_back.callback_manager.CallbackManager import CallbackManager
from executor.packet_factory.lifecycle.initializer.event_initializer import initialize_event_manager
from executor.packet_factory.registry.factory import create_component_by_id,get_component_class_by_id
from executor.packet_factory.logger.logger import LOGGER


def initialize_env(config, agent):
    """
    寻找环境中预留的各个回调，并根据 config 进行插入替换
    :param config: 配置字典
    :param agent: Agent 实例
    :return: 环境实例
    """

    env_type = config.get("env_type")
    if not env_type:
        raise ValueError("[Context] 配置中未指定 'env_type'，请检查配置文件")

    # 获取环境运行模式 (drl | optimization) — 由 Agent 配置提供
    env_mode = config.get(env_type).get("agent", {}).get("mode", "drl")
    
    # todo 后续根据 real 还是 sim 更换创建流程
    env_name = config.get(env_type).get("env_name")
    
    # 创建环境实例，传递 mode 参数
    env = create_component_by_id(env_name, agent, mode=env_mode)
    env: get_component_class_by_id(env_name)

    callback_config = config.get(env_type).get('callback').get('map_callback')

    # headless 模式（如 auto_train_loop.py 直接训练）：跳过可视化器注册，
    # 避免每步 render() 触发 pygame 渲染 + PNG 编码 + clock.tick(fps) 帧率限速（实质是 sleep）
    is_headless = bool(config.get(env_type).get('headless', False))

    # ---------- 创建回调管理器 ----------
    # 无界面训练/benchmark 不应构造 EnvVisualizer，因为其初始化会启动
    # Pygame/SDL，并可能在 macOS 解释器退出时触发 SDL 摇杆清理崩溃。
    callback_manager = CallbackManager(include_visualizer=not is_headless)
    callback_manager.register("load_graph", create_component_by_id(callback_config.get('graph_loader').get('name'),
                                                                   *callback_config.get('graph_loader').get('args')))

    if is_headless:
        # headless 下不注册可视化回调，避免每步渲染和 SDL 初始化。
        LOGGER.info("[Env] headless 模式：跳过可视化器注册")
    else:
        callback_manager.register("initialize_visualizer",
                                  create_component_by_id(callback_config.get('visualizer').get('name'),
                                                         *callback_config.get('visualizer').get('args')))

    # ---------- 创建事件管理器 ----------
    event_manager = initialize_event_manager(config)

    callback_manager.register("event_queue",
                              create_component_by_id(callback_config.get('event_queue').get('name'),
                                                     event_manager))
    env.set_callback_manager(callback_manager)
    
    LOGGER.info(f"[Env] 环境模式设置为：{env_mode}")
    print(f"[Callback] Added to Context...")

    return env
