from django.urls import path
from . import views

urlpatterns = [
    # Tag-based URLs - updated with "/tag/" segment
    path('standard/video/tag/<str:tag_string>.mp4', views.get_video_by_tags, name='video_by_tags'),
    path('standard/cover/tag/<str:tag_string>.webp', views.get_cover_by_tags, name='cover_by_tags'),
    
    # ID-based URLs - updated with "/id/" segment
    path('standard/video/id/<str:video_id>.mp4', views.get_video_by_id, name='video_by_id'),
    path('standard/cover/id/<str:cover_id>.webp', views.get_cover_by_id, name='cover_by_id'),

    # API endpoint for all videos
    path('api/videos/', views.get_all_videos, name='all_videos'),
]