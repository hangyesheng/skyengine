"""
路由注册表：用于统一管理 FastAPI 路由
"""
from typing import Callable, Dict, List, Any
from fastapi import FastAPI
from fastapi.routing import APIRoute


class RouteRegistry:
    """全局路由注册表"""

    _routes: Dict[str, dict] = {}
    # 动态引用：始终指向当前活跃的 BackendCore
    _current_backend_core = None

    @classmethod
    def set_current_backend_core(cls, backend_core):
        """设置当前活跃的 BackendCore（每次切换工厂时更新）"""
        cls._current_backend_core = backend_core

    @classmethod
    def register_route(
        cls,
        path: str,
        method: str = "GET",
        response_class: Any = None,
        **kwargs
    ):
        """
        路由注册装饰器

        Args:
            path: 路由路径
            method: HTTP 方法 (GET, POST, PUT, DELETE 等)
            response_class: 响应类型 (JSONResponse, FileResponse, StreamingResponse 等)
            **kwargs: 其他 FastAPI 路由参数
        """
        def decorator(func: Callable):
            cls._routes[path] = {
                "path": path,
                "method": method,
                "func": func,
                "name": func.__name__,
                "response_class": response_class,
                "kwargs": kwargs
            }
            return func
        return decorator

    @classmethod
    def get_routes(cls) -> Dict[str, dict]:
        """获取所有注册的路由"""
        return cls._routes

    @classmethod
    def clear_routes(cls):
        """清空所有路由"""
        cls._routes = {}

    @classmethod
    def register_to_app(cls, app: FastAPI, prefix: str = ""):
        """
        将注册的路由添加到 FastAPI 应用（仅首次调用时注册）

        路由处理器使用 RouteRegistry._current_backend_core 动态引用，
        确保切换工厂后路由始终指向最新的 BackendCore。

        Args:
            app: FastAPI 实例
            prefix: 路由前缀 (如 "/api")
        """
        # 检查是否已注册过同路径路由（避免重复注册）
        paths_to_register = set(cls._routes.keys())
        already_registered = False
        for route in app.router.routes:
            if isinstance(route, APIRoute):
                route_path = route.path
                if prefix and route_path.startswith(prefix):
                    route_path = route_path[len(prefix):]
                if route_path in paths_to_register:
                    already_registered = True
                    break

        if already_registered:
            # 路由已注册，只需更新 backend_core 引用
            print(f"🔄 路由已存在，仅更新 BackendCore 引用")
            return

        # 首次注册路由
        for path, route_info in cls._routes.items():
            full_path = f"{prefix}{path}"

            route = APIRoute(
                path=full_path,
                endpoint=route_info["func"],
                methods=[route_info["method"]],
                response_class=route_info.get("response_class"),
                **route_info.get("kwargs", {})
            )

            app.router.routes.append(route)
            print(f"✅ 注册路由：{route_info['method']} {full_path}")