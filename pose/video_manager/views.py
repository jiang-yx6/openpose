from django.shortcuts import render
from django.shortcuts import get_object_or_404
from django.http import FileResponse, HttpResponseNotFound, JsonResponse
from django.conf import settings
from rest_framework.views import APIView
from pathlib import Path
from .models import VideoAsset
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import re
import logging

logger = logging.getLogger(__name__)

class VideoByTagsView(APIView):
    @swagger_auto_schema(
        operation_description="Get a video by its tag string",
        manual_parameters=[
            openapi.Parameter(
                name='tag_string',
                in_=openapi.IN_PATH,
                type=openapi.TYPE_STRING,
                description='Tag string to identify the video',
                required=True
            )
        ],
        responses={
            200: "Video file (MP4)",
            404: "Video not found"
        }
    )
    def get(self, request, tag_string):
        """Return a video by its tag string"""

        logger.info(f"Received request for video with tag string: {tag_string}")

        video_asset = get_object_or_404(VideoAsset, tag_string=tag_string)

        if not video_asset:
            logger.warning(f"Video with tag string {tag_string} not found")
        
        file_path = Path(settings.BASE_DIR) / video_asset.original_mp4_path
        
        if file_path.exists():
            return FileResponse(file_path.open('rb'), content_type='video/mp4')
        else:
            return HttpResponseNotFound("Video file not found")


class CoverByTagsView(APIView):
    @swagger_auto_schema(
        operation_description="Get a cover image by its tag string",
        manual_parameters=[
            openapi.Parameter(
                name='tag_string',
                in_=openapi.IN_PATH,
                type=openapi.TYPE_STRING,
                description='Tag string to identify the cover image',
                required=True
            )
        ],
        responses={
            200: "Cover image (WebP)",
            404: "Cover image not found"
        }
    )
    def get(self, request, tag_string):
        """Return a cover image by its tag string"""

        logger.info(f"Received request for cover image with tag string: {tag_string}")

        video_asset = get_object_or_404(VideoAsset, tag_string=tag_string)

        if not video_asset:
            logger.warning(f"Cover image with tag string {tag_string} not found")
        
        file_path = Path(settings.BASE_DIR) / video_asset.original_cover_path
        
        if file_path.exists():
            return FileResponse(file_path.open('rb'), content_type='image/webp')
        else:
            return HttpResponseNotFound("Cover image not found")


class VideoByIdView(APIView):
    @swagger_auto_schema(
        operation_description="Get a video by its numeric ID (format: XX_YY)",
        manual_parameters=[
            openapi.Parameter(
                name='video_id',
                in_=openapi.IN_PATH,
                type=openapi.TYPE_STRING,
                description='Numeric ID in format XX_YY',
                required=True
            )
        ],
        responses={
            200: "Video file (MP4)",
            404: "Video not found or invalid ID format"
        }
    )
    def get(self, request, video_id):
        """Return a video by its numeric ID (format: XX_YY)"""
        parts = video_id.split('_')
        if len(parts) != 2:
            logger.warning(f"Invalid video ID format: {video_id}")
            return HttpResponseNotFound("Invalid video ID format")
        
        try:
            logger.info(f"Received request for video with ID: {video_id}")
            video_asset = VideoAsset.objects.filter(
                numeric_id=video_id
            ).first()
            
            if not video_asset:
                logger.warning(f"Video with ID {video_id} not found")
                return HttpResponseNotFound("Video not found with the specified ID")
                
            file_path = Path(settings.BASE_DIR) / video_asset.original_mp4_path
            
            if file_path.exists():
                logger.info(f"Returning video file: {file_path}")
                return FileResponse(file_path.open('rb'), content_type='video/mp4')
            else:
                logger.warning(f"Video file not found: {file_path}")
                return HttpResponseNotFound("Video file not found")
        except Exception as e:
            logger.error(f"Error retrieving video: {str(e)}", exc_info=True)
            return HttpResponseNotFound(f"Error retrieving video: {str(e)}")


class CoverByIdView(APIView):
    @swagger_auto_schema(
        operation_description="Get a cover image by its numeric ID (format: XX_YY)",
        manual_parameters=[
            openapi.Parameter(
                name='cover_id',
                in_=openapi.IN_PATH,
                type=openapi.TYPE_STRING,
                description='Numeric ID in format XX_YY',
                required=True
            )
        ],
        responses={
            200: "Cover image (WebP)",
            404: "Cover image not found or invalid ID format"
        }
    )
    def get(self, request, cover_id):
        """Return a cover by its numeric ID (format: XX_YY)"""
        parts = cover_id.split('_')
        if len(parts) != 2:
            logger.warning(f"Invalid cover ID format: {cover_id}")
            return HttpResponseNotFound("Invalid cover ID format")
        
        try:
            logger.info(f"Received request for cover with ID: {cover_id}")
            video_asset = VideoAsset.objects.filter(
                numeric_id=cover_id
            ).first()
            
            if not video_asset:
                logger.warning(f"Cover with ID {cover_id} not found")
                return HttpResponseNotFound("Cover not found with the specified ID")
                
            file_path = Path(settings.BASE_DIR) / video_asset.original_cover_path
            
            if file_path.exists():
                logger.info(f"Returning cover file: {file_path}")
                return FileResponse(file_path.open('rb'), content_type='image/webp')
            else:
                logger.warning(f"Cover file not found: {file_path}")
                return HttpResponseNotFound("Cover file not found")
        except Exception as e:
            logger.error(f"Error retrieving cover: {str(e)}", exc_info=True)
            return HttpResponseNotFound(f"Error retrieving cover: {str(e)}")


class AllVideosView(APIView):
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
    def get(self, request):
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
        logger.info(f"Returning {len(video_list)} video assets")
        return JsonResponse({'videos': video_list})


# Keep these function-based views for backward compatibility
def get_video_by_tags(request, tag_string):
    return VideoByTagsView().get(request, tag_string)

def get_cover_by_tags(request, tag_string):
    return CoverByTagsView().get(request, tag_string)

def get_video_by_id(request, video_id):
    return VideoByIdView().get(request, video_id)

def get_cover_by_id(request, cover_id):
    return CoverByIdView().get(request, cover_id)

def get_all_videos(request):
    return AllVideosView().get(request)