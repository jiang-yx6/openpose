from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import VideoUploadSerializer
from .services import VideoProcessingService
from .models import EvalSession, VideoFile
import logging
import time
from django.http import JsonResponse

logger = logging.getLogger(__name__)

# Create your views here.

class VideoUploadView(APIView):
    def post(self, request):
        logger.info(f"收到请求: {request.META.get('HTTP_ORIGIN')}")
        
        serializer = VideoUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            try:
                # 创建会话
                session = EvalSession.objects.create(status='pending')
                logger.info(f"创建会话成功: {session.session_id}")

                # 从base64数据创建文件
                standard_filename = f"standard_{session.session_id}.mp4"
                exercise_filename = f"exercise_{session.session_id}.mp4"
                
                # 转换base64为文件
                standard_file_obj = serializer.base64_to_file(
                    serializer.validated_data['standard'],
                    standard_filename
                )
                exercise_file_obj = serializer.base64_to_file(
                    serializer.validated_data['exercise'],
                    exercise_filename
                )

                # 创建视频文件记录
                standard_file = VideoFile.objects.create(
                    session=session,
                    file=standard_file_obj,
                    video_type='standard'
                )
                exercise_file = VideoFile.objects.create(
                    session=session,
                    file=exercise_file_obj,
                    video_type='exercise'
                )

                # 确保文件已保存
                standard_file.file.close()
                exercise_file.file.close()

                # 启动视频分析和处理
                video_service = VideoProcessingService()
                video_service.process_videos(session.session_id, standard_file.file.path, exercise_file.file.path)
                
                # 返回会话ID和视频URL
                return JsonResponse({
                    'session_id': session.session_id, 
                    'hls_url': f'/media/hls/{session.session_id}/output.m3u8',
                    'standard_video': f'/media/hls/{session.session_id}/standard_annotated.mp4',
                    'exercise_video': f'/media/hls/{session.session_id}/exercise_annotated.mp4'
                }, status=status.HTTP_201_CREATED)

            except Exception as e:
                logger.error(f"处理上传失败: {str(e)}", exc_info=True)
                if 'session' in locals():
                    session.status = 'failed'
                    session.error_message = str(e)
                    session.save()
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
