import cv2
import mediapipe as mp
import numpy as np
import logging
from .models import EvalSession, VideoFile
import os
import subprocess
from .pose_analyzer import PoseDetector, VideoAnalyzer, ActionComparator
from django.conf import settings

logger = logging.getLogger(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

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
        self.pose_processor = PoseProcessor()
        logger.info("初始化视频处理服务")

    def process_videos(self, session_id, standard_video_path, exercise_video_path):
        """使用 DTW 处理视频对比并生成标注后的视频"""
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
            session.frame_data = {'std_frame_data': std_sequence, 'exercise_frame_data': exercise_sequence}
            session.frame_scores = {str(idx): float(score) for idx, score in result['frame_scores']}
            session.status = 'completed'
            session.save()

            logger.info(f"保存分析结果: 总帧数={len(result['frame_scores'])}")
            
            # 4. 生成标注后的视频
            self._generate_annotated_videos(session_id, standard_video_path, exercise_video_path, 
                                           std_sequence, exercise_sequence, result['frame_scores'])

        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}", exc_info=True)
            session = EvalSession.objects.get(pk=session_id)
            session.status = 'failed'
            session.error_message = str(e)
            session.save()
            raise

    def _generate_annotated_videos(self, session_id, standard_video_path, exercise_video_path, 
                                  std_sequence, exercise_sequence, frame_scores):
        """生成标注关键点和得分的视频"""
        try:
            # 创建输出目录
            output_dir = os.path.join(settings.MEDIA_ROOT, 'hls', str(session_id))
            os.makedirs(output_dir, exist_ok=True)
            
            # 标准视频处理
            std_output_path = os.path.join(output_dir, 'standard_annotated.mp4')
            self._process_video_with_annotations(
                standard_video_path, 
                std_output_path, 
                std_sequence, 
                color=(0, 255, 0),  # 绿色
                is_standard=True
            )
            
            # 练习视频处理
            ex_output_path = os.path.join(output_dir, 'exercise_annotated.mp4')
            self._process_video_with_annotations(
                exercise_video_path, 
                ex_output_path, 
                exercise_sequence, 
                frame_scores=frame_scores,
                color=(0, 0, 255)  # 红色
            )
            
            # 生成HLS流
            self._generate_hls_stream(session_id, ex_output_path)
            
            logger.info(f"已生成标注视频: {std_output_path}, {ex_output_path}")
            
        except Exception as e:
            logger.error(f"生成标注视频失败: {str(e)}", exc_info=True)
            raise

    def _process_video_with_annotations(self, input_path, output_path, sequence_data, 
                                       frame_scores=None, color=(0, 255, 0), is_standard=False):
        """处理视频，添加关键点和得分标注"""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {input_path}")
                
            # 获取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            seq_idx = 0  # 序列数据索引
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 每隔2帧处理一次（与process_video中的skip_frames保持一致）
                if seq_idx < len(sequence_data):
                    # 绘制关键点
                    landmarks = sequence_data[seq_idx].get('landmarks', [])
                    for lm in landmarks:
                        if len(lm) >= 3:
                            cv2.circle(frame, (lm[1], lm[2]), 5, color, -1)
                    
                    # 绘制骨架连接
                    self._draw_skeleton(frame, landmarks, color)
                    
                    # 如果有得分数据，显示得分
                    if frame_scores is not None and not is_standard:
                        score = 0
                        if seq_idx < len(frame_scores):
                            score = frame_scores[seq_idx][1]  # (idx, score)
                        
                        cv2.putText(
                            frame,
                            f"Score: {score:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                    
                    seq_idx += 1
                
                # 写入帧
                out.write(frame)
                frame_idx += 1
                
                # 记录进度
                if frame_idx % 100 == 0:
                    logger.info(f"处理视频进度: {frame_idx}/{total_frames} 帧")
            
            # 释放资源
            cap.release()
            out.release()
            
        except Exception as e:
            logger.error(f"处理视频添加标注失败: {str(e)}", exc_info=True)
            raise

    def _draw_skeleton(self, frame, landmarks, color):
        """绘制骨架连接线"""
        # 定义需要连接的关键点对
        connections = [
            (11, 13), (13, 15),  # 左臂
            (12, 14), (14, 16),  # 右臂
            (11, 23), (12, 24),  # 躯干
            (23, 25), (24, 26),  # 左右髋
            (25, 27), (26, 28),  # 左右腿
        ]
        
        try:
            for start_idx, end_idx in connections:
                start_point = None
                end_point = None
                
                for lm in landmarks:
                    if lm[0] == start_idx:
                        start_point = (lm[1], lm[2])
                    elif lm[0] == end_idx:
                        end_point = (lm[1], lm[2])
                
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, color, 2)
        except Exception as e:
            logger.error(f"绘制骨架失败: {str(e)}")

    def _generate_hls_stream(self, session_id, video_path):
        """生成HLS流"""
        try:
            hls_output_path = os.path.join(settings.MEDIA_ROOT, 'hls', str(session_id))
            os.makedirs(hls_output_path, exist_ok=True)

            # FFmpeg 命令
            command = [
                'ffmpeg',
                '-i', video_path,
                '-hls_time', '2',  # 每个片段的时长
                '-hls_list_size', '0',  # 保留所有片段
                '-f', 'hls',
                os.path.join(hls_output_path, 'output.m3u8')  # 输出播放列表
            ]
            logger.info(f"开始生成HLS流: {command}")
            # 启动 FFmpeg 进程并捕获输出
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg 错误: {result.stderr}")
            else:
                logger.info(f"HLS 流处理已启动: {hls_output_path}/output.m3u8")

        except Exception as e:
            logger.error(f"启动 HLS 流处理失败: {str(e)}")
