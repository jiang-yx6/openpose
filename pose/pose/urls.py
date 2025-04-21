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
from evalpose.views import VideoUploadView,TestUploadView,VideoUploadWithReferenceView,FrameScoresView,DeprecatedFrameScoresView
from evalpose import views
from django.views.static import serve
from django.urls import re_path
from .api_docs import urlpatterns as doc_urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload-videos/', VideoUploadView.as_view(), name='upload_videos'),
    path('upload-video/', VideoUploadWithReferenceView.as_view(), name='upload_video_with_reference'),
    # 添加专门的 HLS 文件服务路由
    path('test-upload/', TestUploadView.as_view(), name='test_upload'),
    path('frame-scores/<str:session_id>/', DeprecatedFrameScoresView.as_view(), name='frame_scores'),
    # New URLs for enhanced functionality
    path('scores/frame-scores/<str:session_id>/', views.FrameScoresView.as_view(), name='frame_scores'),
    path('scores/advanced-metrics/<str:session_id>/', views.AdvancedMetricsView.as_view(), name='advanced-metrics'),
    path('scores/exercise-stages/<str:session_id>/', views.ExerciseStagesView.as_view(), name='exercise-stages'),
    path('scores/speed-analysis/<str:session_id>/', views.SpeedAnalysisView.as_view(), name='speed-analysis'),
    path('scores/worst-frames/<str:session_id>/', views.WorstFramesView.as_view(), name='worst-frames'),
    re_path(r'^media/hls/(?P<path>.*)$', serve, {
        'document_root': settings.MEDIA_ROOT + '/hls/',
    }),
    # 添加标准视频文件服务路由
    path('standard/', include('video_manager.urls')),
    # 添加 API 文档路由
    path('api-docs/', include(doc_urls)),
]

# 添加媒体文件的URL配置
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
