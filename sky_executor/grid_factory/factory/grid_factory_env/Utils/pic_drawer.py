from pogema.svg_animation.animation_drawer import SvgSettings, GridHolder, AnimationDrawer
from pogema.svg_animation.svg_objects import Circle, Rectangle
from pogema import AnimationConfig
from typing import List, Union


def draw_svg(env, timeline):
    """
    渲染 Pogema 环境在给定时间戳的 SVG。
    """
    wr = env.grid_config.obs_radius - 1
    if wr > 0:
        obstacles = env.get_obstacles(ignore_borders=False)[wr:-wr, wr:-wr]
    else:
        obstacles = env.get_obstacles(ignore_borders=False)

    full_history = env.decompress_history(env.get_history())

    current_history = []
    for agent_history in full_history:
        last_state = None
        for s in reversed(agent_history):
            if getattr(s, "step", None) is not None and s.step <= timeline:
                last_state = s
                break
        if last_state is None:
            last_state = agent_history[0]
        current_history.append([last_state])

    svg_settings = SvgSettings()
    agents_colors = {
        index: svg_settings.colors[index % len(svg_settings.colors)]
        for index in range(env.grid_config.num_agents)
    }

    grid_holder = GridHolder(
        width=len(obstacles),
        height=len(obstacles[0]),
        obstacles=obstacles,
        episode_length=1,
        history=current_history,
        obs_radius=env.grid_config.obs_radius,
        on_target=env.grid_config.on_target,
        colors=agents_colors,
        config=AnimationConfig(static=True),
        svg_settings=svg_settings
    )

    animation = AnimationDrawer().create_animation(grid_holder)
    return animation.render()

def draw_svg_with_machine(env, machines: List[Union[tuple, object]], timeline=0,
                          shape="circle", color="#1E90FF"):
    """
    在 Pogema SVG 中叠加机器节点。

    Args:
        env: Pogema 环境对象（支持 draw_svg）
        machines: 机器对象列表（可为 [(x, y), ...] 或 [Machine(...), ...]）
        timeline: 时间步
        shape: 绘制形状 ('circle' 或 'rect')
        color: 填充颜色
    """
    # 生成 Pogema 原始地图的 SVG
    svg_str = draw_svg(env, timeline)

    svg_objects = []

    for idx, m in enumerate(machines):
        # 自动提取坐标
        if hasattr(m, "location"):
            x, y = m.location  # Machine 对象
        elif isinstance(m, (tuple, list)) and len(m) == 2:
            x, y = m
        else:
            raise TypeError(f"Unsupported machine type: {type(m)}")

        # Pogema 坐标为 (row, col)
        cx, cy = y + 0.5, x + 0.5  # 注意翻转

        if shape == "circle":
            obj = Circle(cx=cx, cy=cy, r=0.35,
                         fill=color, opacity=0.8,
                         stroke="black", stroke_width=0.02)
        elif shape == "rect":
            obj = Rectangle(x=y + 0.1, y=x + 0.1,
                            width=0.8, height=0.8,
                            fill=color, opacity=0.7,
                            stroke="black", stroke_width=0.02)
        else:
            raise ValueError(f"Unknown shape '{shape}', expected 'circle' or 'rect'.")

        svg_objects.append(obj)

    # 合并机器图层 SVG
    machine_layer = "\n".join(obj.render() for obj in svg_objects)

    # 插入到原始 SVG 的 </svg> 前
    if "</svg>" in svg_str:
        svg_str = svg_str.replace("</svg>", f"{machine_layer}\n</svg>")
    else:
        svg_str += machine_layer

    return svg_str