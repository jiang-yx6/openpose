import cv2
import mediapipe as mp
import numpy as np
import logging
from .models import EvalSession, VideoFile
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import time
import random  # 用于模拟得分
import os
import base64


logger = logging.getLogger(__name__)

class VideoProcessingService:
    def __init__(self):
        self.channel_layer = get_channel_layer()
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose2 = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("初始化视频处理服务")

    def send_score_update(self, session_id, score, status='processing'):
        """
        通过 WebSocket 发送得分更新
        """
        try:
            async_to_sync(self.channel_layer.group_send)(
                f'session_{session_id}',  # 注意这里要和 Consumer 中的 room_group_name 匹配
                {
                    'type': 'score_update',
                    'score': score,
                    'status': status
                }
            )
            logger.info(f"已发送得分更新: session_id={session_id}, score={score}, status={status}")
        except Exception as e:
            logger.error(f"发送得分更新失败: {str(e)}")


    def process_videos(self, session_id, standard_video_path, exercise_video_path):
        """
        模拟视频处理，生成随机评分
        """
        try:
            # 打开两个视频
            cap_standard = cv2.VideoCapture(standard_video_path)
            cap_exercise = cv2.VideoCapture(exercise_video_path)
            
            frame_count = 0
            
            while True:
                # 读取帧
                ret_standard, frame_standard = cap_standard.read()
                ret_exercise, frame_exercise = cap_exercise.read()
                
                if not ret_standard or not ret_exercise:
                    break
                    
                frame_count += 1
                
                # 每5帧发送一次更新
                if frame_count % 5 == 0:
                    # 1. 生成随机评分 (60-100之间)
                    current_score = random.uniform(60, 100)
                    
                    # 发送评分更新
                    async_to_sync(self.channel_layer.group_send)(
                        f'session_{session_id}',
                        {
                            'type': 'score_update',
                            'score': round(current_score, 2),  # 保留两位小数
                            'status': 'processing'
                        }
                    )
                    
                    # 添加一个小延迟，模拟处理时间
                    time.sleep(0.1)
            
            # 处理完成，发送最终评分
            final_score = random.uniform(80, 95)  # 最终评分稍高一些
            async_to_sync(self.channel_layer.group_send)(
                f'session_{session_id}',
                {
                    'type': 'score_update',
                    'score': round(final_score, 2),
                    'status': 'completed'
                }
            )
            
            logger.info(f"视频处理完成: session_id={session_id}, final_score={final_score}")
            
        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}", exc_info=True)
            # 发送错误状态
            async_to_sync(self.channel_layer.group_send)(
                f'session_{session_id}',
                {
                    'type': 'score_update',
                    'score': 0,
                    'status': 'failed',
                    'error': str(e)
                }
            )
            raise
        finally:
            # 释放资源
            cap_standard.release()
            cap_exercise.release()

    # def _simulate_processing(self, session_id):
    #     """
    #     模拟视频处理和评分过程
    #     """
    #     try:
    #         # 模拟处理过程中的得分更新
    #         scores = [30.0, 45.0, 60.0, 75.0, 85.5]
    #         for score in scores:
    #             self.send_score_update(session_id, score, 'processing')
    #             logger.info(f"模拟评分更新: {score}")
    #             time.sleep(2)  # 模拟处理时间

    #         # 最终得分
    #         final_score = random.uniform(80.0, 95.0)
    #         session = EvalSession.objects.get(pk=session_id)
    #         session.score = final_score
    #         session.status = 'completed'
    #         session.save()

    #         # 发送最终得分
    #         self.send_score_update(session_id, final_score, 'completed')
    #         logger.info(f"处理完成，最终评分: {final_score}")

    #     except Exception as e:
    #         logger.error(f"模拟处理失败: {str(e)}", exc_info=True)
    #         raise 