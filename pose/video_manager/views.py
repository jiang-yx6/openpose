from django.shortcuts import render
from django.shortcuts import get_object_or_404
from django.http import FileResponse, HttpResponseNotFound, JsonResponse
from django.conf import settings
import os
from .models import VideoAsset
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import re
import logging

def get_video_by_tags(request, tag_string):
    """Return a video by its tag string"""
    # Find video with matching tag string
    
    video_asset = get_object_or_404(VideoAsset, tag_string=tag_string)
    
    # Construct the actual file path
    file_path = os.path.join(settings.BASE_DIR, video_asset.original_mp4_path)
    
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='video/mp4')
    else:
        return HttpResponseNotFound("Video file not found")

def get_cover_by_tags(request, tag_string):
    """Return a cover image by its tag string"""
    # Find video with matching tag string
    video_asset = get_object_or_404(VideoAsset, tag_string=tag_string)
    
    # Construct the actual file path
    file_path = os.path.join(settings.BASE_DIR, video_asset.original_cover_path)
    
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='image/webp')
    else:
        return HttpResponseNotFound("Cover image not found")

def get_video_by_id(request, video_id):
    """Return a video by its numeric ID (format: XX_YY)"""
    # Parse the ID format
    parts = video_id.split('_')
    if len(parts) != 2:
        return HttpResponseNotFound("Invalid video ID format")
    
    # Try to find by folder ID and file ID
    try:
        folder_id, file_id = parts
        # Find videos in folder with matching folder ID prefix and file ID prefix
        video_asset = VideoAsset.objects.filter(
            numeric_id=video_id
        ).first()
        
        if not video_asset:
            return HttpResponseNotFound("Video not found with the specified ID")
            
        # Construct the actual file path
        file_path = os.path.join(settings.BASE_DIR, video_asset.original_mp4_path)
        
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), content_type='video/mp4')
        else:
            return HttpResponseNotFound("Video file not found")
    except Exception as e:
        return HttpResponseNotFound(f"Error retrieving video: {str(e)}")

def get_cover_by_id(request, cover_id):
    """Return a cover by its numeric ID (format: XX_YY)"""
    # Parse the ID format
    parts = cover_id.split('_')
    if len(parts) != 2:
        return HttpResponseNotFound("Invalid cover ID format")
    
    # Try to find by folder ID and file ID
    try:
        folder_id, file_id = parts
        # Find videos in folder with matching folder ID prefix and file ID prefix
        video_asset = VideoAsset.objects.filter(
            numeric_id=cover_id
        ).first()
        
        if not video_asset:
            return HttpResponseNotFound("Cover not found with the specified ID")
            
        # Get the cover path from the original path by changing extension
        cover_path = os.path.join(settings.BASE_DIR, video_asset.original_cover_path)
        
        # Construct the actual file path
        file_path = os.path.join(settings.BASE_DIR, cover_path)
        
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), content_type='image/webp')
        else:
            return HttpResponseNotFound("Cover file not found")
    except Exception as e:
        return HttpResponseNotFound(f"Error retrieving cover: {str(e)}")
    
@swagger_auto_schema(
    operation_description="Get all video assets with their metadata",
    responses={
        200: openapi.Response(
            description="List of all video assets",
            schema=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'videos': openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                'numeric_id': openapi.Schema(type=openapi.TYPE_STRING),
                                'tag_string': openapi.Schema(type=openapi.TYPE_STRING),
                                'tags': openapi.Schema(type=openapi.TYPE_OBJECT),
                                'video_paths': openapi.Schema(type=openapi.TYPE_OBJECT),
                                'cover_paths': openapi.Schema(type=openapi.TYPE_OBJECT),
                            }
                        )
                    )
                }
            )
        )
    }
)
def get_all_videos(request):
    """Return all video assets with their tags, IDs and file paths"""
    videos = VideoAsset.objects.all()
    
    video_list = []
    for video in videos:
        video_data = {
            'numeric_id': video.numeric_id,
            'tag_string': video.tag_string,
            'tags': {
                'tag1': video.tag1,
                'tag2': video.tag2,
                'tag3': video.tag3,
                'tag4': video.tag4,
                'tag5': video.tag5,
            },
            'video_paths': video.mp4_path,
            'cover_paths': video.cover_path,
        }
        video_list.append(video_data)
    
    return JsonResponse({'videos': video_list})