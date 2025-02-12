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
        处理上传的视频文件
        """
        try:
            # 获取会话
            session = EvalSession.objects.get(pk=session_id)
            session.status = 'processing'
            session.save()

            # 检查文件是否存在
            if not os.path.exists(standard_video_path):
                raise FileNotFoundError(f"标准视频文件不存在: {standard_video_path}")
            if not os.path.exists(exercise_video_path):
                raise FileNotFoundError(f"练习视频文件不存在: {exercise_video_path}")

            logger.info(f"开始处理视频: standard={standard_video_path}, exercise={exercise_video_path}")

            cap1 = cv2.VideoCapture(exercise_video_path)
            cap2 = cv2.VideoCapture(standard_video_path)

            total_similarity = 0
            frame_count = 0

            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()

                if not ret1 or not ret2:
                    break

                # 处理帧并获取相似度
                similarity, results1, results2 = self.process_frame(frame1, frame2)
                
                # 更新总相似度
                total_similarity += similarity
                frame_count += 1

                # 每25帧发送一次更新
                if frame_count % 25 == 0:
                    self.send_score_update(
                        session_id, 
                        similarity * 100,  # 转换为百分比
                        'processing'
                    )

            # 计算最终得分
            final_score = (total_similarity / frame_count) * 100 if frame_count > 0 else 0

            # 更新会话状态
            session.score = final_score
            session.status = 'completed'
            session.save()

            # 发送最终得分
            self.send_score_update(session_id, final_score, 'completed')
            logger.info(f"处理完成，最终评分: {final_score}")

            cap1.release()
            cap2.release()

        except Exception as e:
            logger.error(f"视频处理发生错误: {str(e)}")
            if session:
                session.status = 'failed'
                session.error_message = str(e)
                session.save()
            raise

    def _simulate_processing(self, session_id):
        """
        模拟视频处理和评分过程
        """
        try:
            # 模拟处理过程中的得分更新
            scores = [30.0, 45.0, 60.0, 75.0, 85.5]
            for score in scores:
                self.send_score_update(session_id, score, 'processing')
                logger.info(f"模拟评分更新: {score}")
                time.sleep(2)  # 模拟处理时间

            # 最终得分
            final_score = random.uniform(80.0, 95.0)
            session = EvalSession.objects.get(pk=session_id)
            session.score = final_score
            session.status = 'completed'
            session.save()

            # 发送最终得分
            self.send_score_update(session_id, final_score, 'completed')
            logger.info(f"处理完成，最终评分: {final_score}")

        except Exception as e:
            logger.error(f"模拟处理失败: {str(e)}", exc_info=True)
            raise 