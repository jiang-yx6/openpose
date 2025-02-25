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
import cv2
import mediapipe as mp
import time
import random
import matplotlib.pyplot as plt
from asgiref.sync import async_to_sync
from django.conf import settings
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union, Mapping
from .pose_analyzer import PoseDetector, VideoAnalyzer,ActionComparator

logger = logging.getLogger(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

@dataclass
class DrawingSpec:
    color: Tuple[int, int, int] = (224, 224, 224)
    thickness: int = 2
    circle_radius: int = 2

class PoseProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_frame(self, frame, draw_landmarks=True):
        """处理单帧并返回处理后的帧和姿态结果"""
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(frame_rgb)
        frame.flags.writeable = True
        
        if draw_landmarks and pose_results.pose_landmarks:
            self.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
        return frame, pose_results
        
    def draw_landmarks(self, image, landmark_list, connections=None,
                      color=(0, 255, 0), thickness=2):
        """简化版的关键点绘制函数"""
        if not landmark_list:
            return
            
        h, w, _ = image.shape
        for landmark in landmark_list.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 5, color, thickness)
            
        if connections:
            for connection in connections:
                start_point = landmark_list.landmark[connection[0]]
                end_point = landmark_list.landmark[connection[1]]
                
                start_x = int(start_point.x * w)
                start_y = int(start_point.y * h)
                end_x = int(end_point.x * w)
                end_y = int(end_point.y * h)
                
                cv2.line(image, (start_x, start_y), 
                        (end_x, end_y), color, thickness)

class VideoProcessingService:
    def __init__(self):
        self.channel_layer = get_channel_layer()
        self.pose_processor = PoseProcessor()
        logger.info("初始化视频处理服务")

    def process_frame_for_stream(self, frame):
        """处理流媒体帧并绘制关键点"""
        try:
            # 处理帧并获取姿态结果
            processed_frame, pose_results = self.pose_processor.process_frame(
                frame, 
                draw_landmarks=True  # 确保绘制关键点
            )
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"处理流媒体帧失败: {str(e)}")
            return frame  # 如果处理失败，返回原始帧

    def process_videos(self, session_id, standard_video_path, exercise_video_path):
        """使用 DTW 处理视频对比"""
        try:
            logger.info(f"开始处理视频: session_id={session_id}")
            
            # 1. DTW 分析
            analyzer = VideoAnalyzer()
            
            # 获取所有帧数据
            std_sequence = analyzer.process_video(standard_video_path, skip_frames=2) 
            
            exercise_sequence = analyzer.process_video(exercise_video_path, skip_frames=2)
            
            # 2. 比较序列
            comparator = ActionComparator(std_sequence, exercise_sequence)
            result = comparator.compare_sequences()
            
            logger.info(f"DTW分析完成: {result['similarity_score']}")

            # 3. 保存结果
            session = EvalSession.objects.get(pk=session_id)
            session.dtw_distance = float(result['dtw_distance'])
            session.similarity_score = float(result['similarity_score'] * 100)
            
            # 保存所有帧的评分
            frame_scores = {str(idx): float(score) for idx, score in result['frame_scores']}
            session.frame_scores = frame_scores
            
            logger.info(f"保存分析结果: 总帧数={len(frame_scores)}")
            session.status = 'completed'
            session.save()
            
            logger.info(f"保存分析结果: 总帧数={len(frame_scores)}")

        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}", exc_info=True)
            session = EvalSession.objects.get(pk=session_id)
            session.status = 'failed'
            session.error_message = str(e)
            session.save()
            raise


    def _calculate_similarity(self, landmarks_standard, landmarks_exercise):
        """计算两组关键点的相似度"""
        if not landmarks_standard or not landmarks_exercise:
            return 0
            
        try:
            total_similarity = 0
            valid_points = 0
            
            for i in range(33):  # MediaPipe 姿态估计的33个关键点
                std_point = landmarks_standard.landmark[i]
                ex_point = landmarks_exercise.landmark[i]
                
                # 检查关键点的可见性
                if (hasattr(std_point, 'visibility') and std_point.visibility < 0.5) or \
                   (hasattr(ex_point, 'visibility') and ex_point.visibility < 0.5):
                    continue
                
                # 计算欧氏距离
                distance = ((std_point.x - ex_point.x) ** 2 + 
                          (std_point.y - ex_point.y) ** 2) ** 0.5
                          
                # 转换为相似度分数
                similarity = max(0, 1 - distance)
                total_similarity += similarity
                valid_points += 1
            
            # 确保至少有一个有效的关键点
            if valid_points == 0:
                logger.warning("没有有效的关键点用于计算相似度")
                return 0
                
            return total_similarity / valid_points
            
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return 0