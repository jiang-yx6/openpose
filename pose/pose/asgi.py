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
from channels.security.websocket import AllowedHostsOriginValidator
from evalpose.routing import websocket_urlpatterns

# 设置环境变量
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pose.settings')
django.setup()  # 添加这行

# 在设置环境变量后再导入

# 初始化 Django ASGI application
django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,  # 处理普通的 HTTP 请求（如视频上传）
    "websocket": AllowedHostsOriginValidator(  # 处理 WebSocket 连接（实时评分更新）
        AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        )
    ),
})
