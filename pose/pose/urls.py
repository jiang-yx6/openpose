"""
URL configuration for pose project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from evalpose.views import VideoUploadView,TestUploadView,FrameScoresView,VideoUploadWithReferenceView
from django.views.static import serve
from django.urls import re_path
from .api_docs import urlpatterns as doc_urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload-videos/', VideoUploadView.as_view(), name='upload_videos'),
    path('upload-video/', VideoUploadWithReferenceView.as_view(), name='upload_video_with_reference'),
    # 添加专门的 HLS 文件服务路由
    path('test-upload/', TestUploadView.as_view(), name='test_upload'),
    path('frame-scores/<str:session_id>/', FrameScoresView.as_view(), name='frame_scores'),
    re_path(r'^media/hls/(?P<path>.*)$', serve, {
        'document_root': settings.MEDIA_ROOT + '/hls/',
    }),
    # 添加标准视频文件服务路由
    path('standard/', include('video_manager.urls')),
    # 添加 API 文档路由
    path('api-docs/', include(doc_urls)),
]

# 添加媒体文件的URL配置
# Uncomment this line to serve static files in development
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# Keep the media URL configuration too
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
