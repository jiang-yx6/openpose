from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import VideoUploadSerializer
# from .services import VideoProcessingService
from .models import EvalSession, VideoFile
import logging
import time
from django.http import JsonResponse
from django.conf import settings
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from pathlib import Path
from video_manager.models import VideoAsset
from .service_factory import get_video_processing_service
from django.utils.deprecation import RemovedInNextVersionWarning
import warnings

logger = logging.getLogger(__name__)

# Create your views here.

class VideoUploadView(APIView):
    # Decrepated: This view is now replaced by VideoUploadWithReferenceView and may cause incorrect results and confusive errors.
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
        },
        deprecated=True
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
                video_service = get_video_processing_service()
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
                    },
                    'frame_scores': process_result['frame_scores'],
                    'advanced_metrics': process_result.get('advanced_metrics', {}),
                    'action_stages': process_result.get('action_stages', []),
                    'lowest_score_frames': process_result.get('lowest_score_frames', []),
                    'standard_video_info': {
                        'numeric_id': standard_file.numeric_id,
                        'tag_string': standard_file.tag_string,
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

class VideoUploadWithReferenceView(APIView):
    @swagger_auto_schema(
        operation_description="Upload exercise video and reference a standard video by its numeric_id",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['exercise', 'standard_numeric_id'],
            properties={
                'exercise': openapi.Schema(
                    type=openapi.TYPE_STRING, 
                    description='Base64 encoded exercise video'
                ),
                'standard_numeric_id': openapi.Schema(
                    type=openapi.TYPE_STRING, 
                    description='Numeric ID of the standard video to reference'
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
            400: "Invalid input or standard video not found",
            500: "Server error during processing"
        }
    )
    def post(self, request):
        logger.info(f"收到带参考视频ID的请求: {request.META.get('HTTP_ORIGIN')}")
        
        # Validate input data
        if 'exercise' not in request.data:
            return Response({'error': 'Missing exercise video'}, status=status.HTTP_400_BAD_REQUEST)
            
        if 'standard_numeric_id' not in request.data:
            return Response({'error': 'Missing standard_numeric_id'}, status=status.HTTP_400_BAD_REQUEST)
        
        standard_numeric_id = request.data['standard_numeric_id']
        
        # Look up the standard video by numeric_id
        standard_video = VideoAsset.objects.filter(numeric_id=standard_numeric_id).first()
        if not standard_video:
            logger.warning(f"找不到指定ID的标准视频: {standard_numeric_id}")
            return Response({'error': f'Standard video with ID {standard_numeric_id} not found'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        # Get path to the standard video file
        standard_video_path = Path(settings.BASE_DIR) / standard_video.original_mp4_path
        if not standard_video_path.exists():
            logger.warning(f"标准视频文件不存在: {standard_video_path}")
            return Response({'error': 'Standard video file not found on server'}, 
                          status=status.HTTP_400_BAD_REQUEST)
            
        try:
            # Create session
            session = EvalSession.objects.create(status='pending')
            logger.info(f"创建会话成功: {session.session_id}")

            # Create exercise file from base64 data
            exercise_filename = f"exercise_{session.session_id}.mp4"
            
            # Convert base64 to file
            serializer = VideoUploadSerializer()  # Create serializer instance for helper methods
            exercise_file_obj = serializer.base64_to_file(request.data['exercise'], exercise_filename)

            # Create video file record for exercise
            exercise_file = VideoFile.objects.create(
                session=session,
                file=exercise_file_obj,
                video_type='exercise'
            )

            # Ensure file is saved
            exercise_file.file.close()

            # Start video analysis and processing with referenced standard video
            video_service = get_video_processing_service()
            process_result = video_service.process_videos(
                session.session_id, 
                str(standard_video_path),  # Use the path to the referenced standard video
                exercise_file.file.path
            )
            
            # Check processing results
            if not process_result['hls_success']:
                logger.warning(f"HLS 转换未完全成功: standard={process_result['standard_hls']}, exercise={process_result['exercise_hls']}")
            
            # Return session ID and video URLs
            response_data = {
                'session_id': session.session_id,
                'standard_video_hls': f'/media/hls/{session.session_id}/standard.m3u8',
                'exercise_video_hls': f'/media/hls/{session.session_id}/exercise.m3u8',
                'overlap_video_hls': f'/media/hls/{session.session_id}/overlap.m3u8',
                'exercise_worst_frames': [f'/media/hls/{session.session_id}/patient_frame_1.jpg',
                                          f'/media/hls/{session.session_id}/patient_frame_2.jpg',
                                          f'/media/hls/{session.session_id}/patient_frame_3.jpg'],
                'frame_scores': process_result['frame_scores'],
                'standard_video_info': {
                    'numeric_id': standard_video.numeric_id,
                    'tag_string': standard_video.tag_string,
                },
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

class AdvancedMetricsView(APIView):
    @swagger_auto_schema(
        operation_description="Get advanced metrics for a session",
        manual_parameters=[
            openapi.Parameter(
                name='session_id',
                in_=openapi.IN_PATH,
                type=openapi.TYPE_STRING,
                description='Session ID to get advanced metrics for',
                required=True
            )
        ],
        responses={
            200: openapi.Response(
                description="Advanced metrics data",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'alignment_angle': openapi.Schema(type=openapi.TYPE_OBJECT),
                        'time_variance': openapi.Schema(type=openapi.TYPE_OBJECT),
                        'alignment_ratio': openapi.Schema(type=openapi.TYPE_NUMBER),
                        'speed_variation': openapi.Schema(type=openapi.TYPE_OBJECT),
                        'overall_scores': openapi.Schema(type=openapi.TYPE_OBJECT),
                    }
                )
            ),
            404: "Session not found",
            500: "Server error"
        }
    )
    def get(self, request, session_id):
        service = get_video_processing_service(use_modern=True)
        try:
            # Add debug logging
            from .models import EvalSession
            session = EvalSession.objects.get(pk=session_id)
            logger.info(f"Session {session_id} frame_data keys: {list(session.frame_data.keys() if session.frame_data else [])}")
            
            metrics = service.get_advanced_metrics(session_id)
            logger.info(f"Advanced metrics for session {session_id}: {metrics}")
            return Response(metrics, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching advanced metrics: {str(e)}")
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ExerciseStagesView(APIView):
    @swagger_auto_schema(
        operation_description="Get exercise stages for a session",
        manual_parameters=[
            openapi.Parameter(
                name='session_id',
                in_=openapi.IN_PATH,
                type=openapi.TYPE_STRING,
                description='Session ID',
                required=True
            )
        ],
        responses={
            200: openapi.Response(
                description="Exercise stages data",
                schema=openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_INTEGER)
                    )
                )
            ),
            404: "Session not found",
            500: "Server error"
        }
    )
    def get(self, request, session_id):
        service = get_video_processing_service(use_modern=True)
        try:
            stages = service.detect_exercise_stages(session_id)
            return Response(stages, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SpeedAnalysisView(APIView):
    @swagger_auto_schema(
        operation_description="Get speed analysis for a session",
        manual_parameters=[
            openapi.Parameter(
                name='session_id',
                in_=openapi.IN_PATH,
                type=openapi.TYPE_STRING,
                description='Session ID',
                required=True
            )
        ],
        responses={
            200: openapi.Response(
                description="Speed analysis data",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'slow_segments': openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Schema(
                                type=openapi.TYPE_ARRAY,
                                items=openapi.Schema(type=openapi.TYPE_INTEGER)
                            )
                        ),
                        'fast_segments': openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Schema(
                                type=openapi.TYPE_ARRAY,
                                items=openapi.Schema(type=openapi.TYPE_INTEGER)
                            )
                        ),
                    }
                )
            ),
            404: "Session not found",
            500: "Server error"
        }
    )
    def get(self, request, session_id):
        service = get_video_processing_service(use_modern=True)
        try:
            analysis = service.get_speed_analysis(session_id)
            return Response(analysis, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class WorstFramesView(APIView):
    @swagger_auto_schema(
        operation_description="Get worst performing frames for a session",
        manual_parameters=[
            openapi.Parameter(
                name='session_id',
                in_=openapi.IN_PATH,
                type=openapi.TYPE_STRING,
                description='Session ID',
                required=True
            ),
            openapi.Parameter(
                name='max_frames',
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_INTEGER,
                description='Maximum number of frames to return',
                required=False,
                default=3
            )
        ],
        responses={
            200: openapi.Response(
                description="Worst frames data",
                schema=openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_NUMBER)
                    )
                )
            ),
            404: "Session not found",
            500: "Server error"
        }
    )
    def get(self, request, session_id):
        max_frames = int(request.query_params.get('max_frames', 3))
        service = get_video_processing_service(use_modern=True)
        
        try:
            worst_frames = service.get_worst_frames(session_id, max_frames)
            return Response(worst_frames, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DeprecatedFrameScoresView(FrameScoresView):
    @swagger_auto_schema(
        operation_description="Get frame scores for a session (DEPRECATED: Use /scores/frame-scores/ instead)",
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
        },
        deprecated=True
    )
    def get(self, request, session_id):
        # You can optionally add a warning in the response
        warnings.warn(
            "The /frame-scores/ endpoint is deprecated. Please use /scores/frame-scores/ instead.",
            RemovedInNextVersionWarning,
            stacklevel=2
        )
        return super().get(request, session_id)
