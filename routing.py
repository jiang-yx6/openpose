from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # 匹配前端的连接路径
    re_path(r'ws/(?P<session_id>[\w-]+)/(?P<type>\w+)/$', consumers.CombinedConsumer.as_asgi()),
] 