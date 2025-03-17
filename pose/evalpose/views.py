from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import VideoUploadSerializer
from .services import VideoProcessingService
from .models import EvalSession, VideoFile
import logging
import time
from django.http import JsonResponse
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

logger = logging.getLogger(__name__)

# Create your views here.

class VideoUploadView(APIView):
    @swagger_auto_schema(
        operation_description="Upload standard and exercise videos for analysis",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['standard', 'exercise'],
            properties={
                'standard': openapi.Schema(
                    type=openapi.TYPE_STRING, 
                    description='Base64 encoded standard video'
                ),
                'exercise': openapi.Schema(
                    type=openapi.TYPE_STRING, 
                    description='Base64 encoded exercise video'
                ),
            }
        ),
        responses={
            201: openapi.Response(
                description="Successfully processed videos",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'session_id': openapi.Schema(type=openapi.TYPE_STRING),
                        'standard_video_hls': openapi.Schema(type=openapi.TYPE_STRING),
                        'exercise_video_hls': openapi.Schema(type=openapi.TYPE_STRING),
                        'overlap_video_hls': openapi.Schema(type=openapi.TYPE_STRING),
                        'exercise_worst_frames': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING)),
                        'frame_scores': openapi.Schema(type=openapi.TYPE_OBJECT),
                        'processing_status': openapi.Schema(type=openapi.TYPE_OBJECT),
                    }
                )
            ),
            400: "Invalid input data",
            500: "Server error during processing"
        }
    )
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
                standard_file_obj = serializer.base64_to_file(serializer.validated_data['standard'],standard_filename)
                exercise_file_obj = serializer.base64_to_file(serializer.validated_data['exercise'],exercise_filename)

                # 创建视频文件记录
                standard_file = VideoFile.objects.create(session=session,file=standard_file_obj,video_type='standard')
                exercise_file = VideoFile.objects.create(session=session,file=exercise_file_obj,video_type='exercise')

                # 确保文件已保存
                standard_file.file.close()
                exercise_file.file.close()

                # 启动视频分析和处理
                video_service = VideoProcessingService()
                process_result = video_service.process_videos(
                    session.session_id, 
                    standard_file.file.path, 
                    exercise_file.file.path
                )
                
                # 检查处理结果
                if not process_result['hls_success']:
                    logger.warning(f"HLS 转换未完全成功: standard={process_result['standard_hls']}, exercise={process_result['exercise_hls']}")
                
                # 返回会话ID和视频URL
                response_data = {
                    'session_id': session.session_id,
                    'standard_video_hls': f'/media/hls/{session.session_id}/standard.m3u8',
                    'exercise_video_hls': f'/media/hls/{session.session_id}/exercise.m3u8',
                    'overlap_video_hls': f'/media/hls/{session.session_id}/overlap.m3u8',
                    'exercise_worst_frames': [f'/media/hls/{session.session_id}/patient_frame_1.jpg',
                                              f'/media/hls/{session.session_id}/patient_frame_2.jpg',
                                              f'/media/hls/{session.session_id}/patient_frame_3.jpg'],
                    'frame_scores': process_result['frame_scores'],
                    'processing_status': {
                        'dtw_success': process_result['dtw_success'],
                        'hls_success': process_result['hls_success'],
                        'standard_hls': process_result['standard_hls'],
                        'exercise_hls': process_result['exercise_hls'],
                        'overlap_hls': process_result['overlap_hls']
                    }
                }
                
                return JsonResponse(response_data, status=status.HTTP_201_CREATED)

            except Exception as e:
                logger.error(f"处理上传失败: {str(e)}", exc_info=True)
                if 'session' in locals():
                    session.status = 'failed'
                    session.error_message = str(e)
                    session.save()
                return Response({
                    'error': str(e),
                    'status': 'failed'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class FrameScoresView(APIView):
    @swagger_auto_schema(
        operation_description="Get frame scores for a session",
        manual_parameters=[
            openapi.Parameter(
                name='session_id',
                in_=openapi.IN_PATH,
                type=openapi.TYPE_STRING,
                description='Session ID to get frame scores for',
                required=True
            )
        ],
        responses={
            200: openapi.Response(
                description="Frame scores data",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'status': openapi.Schema(type=openapi.TYPE_STRING),
                        'session_id': openapi.Schema(type=openapi.TYPE_STRING),
                        'frame_scores': openapi.Schema(type=openapi.TYPE_OBJECT),
                    }
                )
            ),
            400: "Session not completed or not found"
        }
    )
    def get(self, request, session_id):
        session = EvalSession.objects.get(session_id=session_id)
        if session.status != 'completed':
            return Response({
                'error': '会话未完成',
                'status': 'failed'
            }, status=status.HTTP_400_BAD_REQUEST)
        frame_scores = session.frame_scores
        response_data = {
            "status": "success",
            "session_id": session_id,
            "frame_scores": frame_scores
        }
        return Response(response_data, status=status.HTTP_200_OK)        


class TestUploadView(APIView):
    @swagger_auto_schema(
        operation_description="Test endpoint for video upload",
        responses={
            201: openapi.Response(
                description="Test response",
                schema=openapi.Schema(type=openapi.TYPE_OBJECT)
            )
        }
    )
    def post(self, request):
        # # 返回会话ID和视频URL
        # logger.info(f"收到请求: {request.META.get('HTTP_ORIGIN')}")

        # response_data = {
        #     'session_id': session.session_id,
        #     'standard_video_hls': f'/media/hls/{session.session_id}/standard.m3u8',
        #     'exercise_video_hls': f'/media/hls/{session.session_id}/exercise.m3u8',
        #     'overlap_video_hls': f'/media/hls/{session.session_id}/overlap.m3u8',
        #     'exercise_worst_frames': [f'/media/hls/{session.session_id}/patient_frame_1.jpg',
        #                                 f'/media/hls/{session.session_id}/patient_frame_2.jpg',
        #                                 f'/media/hls/{session.session_id}/patient_frame_3.jpg'],
        #     'processing_status': {
        #         'dtw_success': process_result['dtw_success'],
        #         'hls_success': process_result['hls_success'],
        #         'standard_hls': process_result['standard_hls'],
        #         'exercise_hls': process_result['exercise_hls'],
        #         'overlap_hls': process_result['overlap_hls']
        #     }
        # }
        response_data = {}
        return JsonResponse(response_data, status=status.HTTP_201_CREATED)
