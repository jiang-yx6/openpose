# pose/evalpose/routing.py
from django.urls import re_path
from . import consumers  # 导入你的消费者

websocket_urlpatterns = [
    re_path(r'ws/(?P<session_id>[\w-]+)/(?P<type>\w+)/$', consumers.VideoAnalysisConsumer.as_asgi()),
]