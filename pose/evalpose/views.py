from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import VideoUploadSerializer, SessionSerializer
# from .services import VideoProcessingService
from .models import EvalSession, VideoFile
import threading
import logging
from django.core.files.base import ContentFile
import os

logger = logging.getLogger(__name__)

# Create your views here.

class VideoUploadView(APIView):
    def post(self, request):
        logger.info(f"收到请求: {request.META.get('HTTP_ORIGIN')}")
        logger.info(f"请求方法: {request.method}")
        logger.info(f"请求头: {request.headers}")
        
        serializer = VideoUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            try:
                # 创建会话
                session = EvalSession.objects.create(status='pending')
                logger.info(f"创建会话成功: {session.session_id}")

                # 先保存视频文件
                standard_video = serializer.validated_data['standardVideo']
                exercise_video = serializer.validated_data['exerciseVideo']

                # 创建视频文件记录
                standard_file = VideoFile.objects.create(
                    session=session,
                    file=standard_video,
                    video_type='standard'
                )
                exercise_file = VideoFile.objects.create(
                    session=session,
                    file=exercise_video,
                    video_type='exercise'
                )

                # 确保文件已保存
                standard_file.file.close()
                exercise_file.file.close()
                
                # 立即返回会话ID
                response_serializer = SessionSerializer(session)
                return Response(response_serializer.data, status=status.HTTP_201_CREATED)
                
            except Exception as e:
                logger.error(f"创建会话失败: {str(e)}", exc_info=True)
                return Response(
                    {'error': str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class HealthCheckView(APIView):
    def get(self, request):
        return Response({"status": "ok"})
