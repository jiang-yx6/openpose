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
