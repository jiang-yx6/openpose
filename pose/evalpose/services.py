import cv2
import mediapipe as mp
import numpy as np
import logging
from .models import EvalSession, VideoFile
import os
import subprocess
from .pose_analyzer import VideoAnalyzer, ActionComparator
from django.conf import settings
from concurrent.futures import ThreadPoolExecutor
from .exceptions import ApiErrorHandler, FullBodyNotVisibleError

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
        self.hls_ouput_path = None
        logger.info("初始化视频处理服务")

    def process_videos(self, session_id, standard_video_path, exercise_video_path):
        """使用 DTW 处理视频对比并生成标注后的视频"""
        result = {
            'dtw_success': False,
            'hls_success': False,
            'standard_hls': False,
            'exercise_hls': False,
            'overlap_hls': False,
            'frame_scores': []
        }
        
        self.hls_ouput_path = os.path.join(settings.MEDIA_ROOT, 'hls', str(session_id))
        try:
            logger.info(f"开始处理视频: session_id={session_id}")
            
            # 1. DTW 分析
            analyzer = VideoAnalyzer()
            
            # 获取所有帧数据
            std_sequence = analyzer.process_video(standard_video_path, skip_frames=2) 
            exe_sequence = analyzer.process_video(exercise_video_path, skip_frames=2)

            # 2. 比较序列
            comparator = ActionComparator(std_sequence, exe_sequence)
            dtw_result = comparator.compare_sequences()
            result['dtw_success'] = True
            
            logger.info(f"DTW分析完成: {dtw_result['similarity_score']}")

            # 3. 保存结果
            session = EvalSession.objects.get(pk=session_id)
            session.dtw_distance = float(dtw_result['dtw_distance'])
            session.similarity_score = float(dtw_result['similarity_score'] * 100)
            session.frame_data = {'std_frame_data': std_sequence, 'exercise_frame_data': exe_sequence}
            session.frame_scores = {str(idx): float(score) for idx, score in dtw_result['frame_scores']}
            result['frame_scores'] = session.frame_scores
            session.status = 'completed'
            session.save()

            # 4. 生成标注后的视频并转换为HLS
            output_dir = os.path.join(settings.MEDIA_ROOT, 'hls', str(session_id))
            os.makedirs(output_dir, exist_ok=True)
            
            std_output_path = os.path.join(output_dir, 'standard_annotated.mp4')
            ex_output_path = os.path.join(output_dir, 'exercise_annotated.mp4')
            overlap_output_path = os.path.join(output_dir, 'overlap_annotated.mp4')


            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(self._process_video_with_annotations, standard_video_path, std_output_path, std_sequence, color=(0, 255, 0), is_standard=True),
                    executor.submit(self._process_video_with_annotations, exercise_video_path, ex_output_path, exe_sequence, frame_scores=dtw_result['frame_scores'], color=(0, 0, 255)),
                    executor.submit(analyzer._process_overlap_video, std_sequence, exe_sequence, dtw_result, overlap_output_path, exercise_video_path)
                ]
                for future in futures:
                    future.result()
            # 处理标准视频
            # self._process_video_with_annotations(
            #     standard_video_path, 
            #     std_output_path, 
            #     std_sequence, 
            #     color=(0, 255, 0),
            #     is_standard=True
            # )
            # # 处理练习视频
            # self._process_video_with_annotations(
            #     exercise_video_path, 
            #     ex_output_path, 
            #     exe_sequence, 
            #     frame_scores=dtw_result['frame_scores'],
            #     color=(0, 0, 255)
            # )
            
            # analyzer._process_overlap_video(
            #     std_sequence,
            #     exe_sequence,
            #     dtw_result,
            #     overlap_output_path,
            #     exercise_video_path               
            # )

            # 生成HLS流
            result['overlap_hls'] = self._generate_hls_stream(session_id, overlap_output_path, 'overlap')
            result['standard_hls'] = self._generate_hls_stream(session_id, std_output_path, 'standard')
            result['exercise_hls'] = self._generate_hls_stream(session_id, ex_output_path, 'exercise')
            result['hls_success'] = result['standard_hls'] and result['exercise_hls'] and  result['overlap_hls']
            
            logger.info(f"视频处理完成，HLS转换状态: {result}")
            return result

        except ValueError as e:
            # Check if this is the inhomogeneous shape error
            if ApiErrorHandler.is_inhomogeneous_shape_error(e):
                raise FullBodyNotVisibleError() from e
            raise
        except IndexError as e:
            # Check if this is a list index out of range error
            if "list index out of range" in str(e):
                raise VideoLengthMismatchError() from e
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

    def _generate_hls_stream(self, session_id, video_path, video_type='exercise'):
        """生成HLS流"""
        try:
            hls_output_path = os.path.join(settings.MEDIA_ROOT, 'hls', str(session_id))
            os.makedirs(hls_output_path, exist_ok=True)

            # 根据视频类型设置输出文件名
            output_filename = f"{video_type}.m3u8"

            # FFmpeg 命令
            command = [
                'ffmpeg',
                '-i', video_path,
                '-hls_time', '2',  # 每个片段的时长
                '-hls_list_size', '0',  # 保留所有片段
                '-hls_segment_filename', os.path.join(hls_output_path, f"{video_type}_%03d.ts"),  # 片段文件名格式
                '-f', 'hls',
                os.path.join(hls_output_path, output_filename)  # 输出播放列表
            ]
            logger.info(f"开始生成{video_type}视频的HLS流: {' '.join(command)}")
            # 使用 Popen 来运行 FFmpeg，并等待完成
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 实时获取输出
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.debug(f"FFmpeg 输出: {output.strip()}")
            
            # 获取最终结果
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"FFmpeg 错误: {stderr}")
                return False
            else:
                logger.info(f"HLS 流处理完成: {hls_output_path}/{output_filename}")
                return True
           
        except Exception as e:
            logger.error(f"启动 HLS 流处理失败: {str(e)}")

