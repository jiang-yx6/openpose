"""
ASGI config for pose project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
import django
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

# 设置 Django 设置模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pose.settings')

# 初始化 Django
django.setup()

# 在 Django 设置完成后导入路由
from evalpose import routing  # 导入你的路由配置

# 初始化 Django ASGI application
django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,  # 处理普通的 HTTP 请求（如视频上传）
    "websocket": AuthMiddlewareStack(  # 处理 WebSocket 连接（实时评分更新）
        URLRouter(
            routing.websocket_urlpatterns  # 使用你的 WebSocket 路由
        )
    ),
})
