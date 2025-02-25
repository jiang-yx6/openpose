import cv2
from av import VideoFrame
from aiortc import MediaStreamTrack
import fractions
import logging
from .models import EvalSession
from channels.db import database_sync_to_async
import asyncio
from channels.layers import get_channel_layer
import base64

logger = logging.getLogger(__name__)

class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, video_path, video_service, session_id, type):
        super().__init__()
        self.video_path = video_path
        self.type = type
        self.cap = cv2.VideoCapture(video_path)
        self.video_service = video_service
        
        self.frame_count = 0
        self.ended = False
        self.session_id = session_id
        
        # 异步获取帧评分数据
        self.frame_scores = {}  # 初始化为空字典
        self.scores_initialized = False  # 添加标志
        self.max_retries = 60  # 增加等待时间到60秒
        self.retry_interval = 2  # 每次等待2秒
        
        # 记录实际播放的帧的评分
        self.played_frame_scores = []
        self.channel_layer = get_channel_layer()

        self.keyframe_interval = 30  # 每30帧发送一次关键帧
        self.last_frame = None  # 存储最近的帧
        self.last_keypoints = None  # 存储最近的关键点
    @database_sync_to_async
    def _get_frame_scores(self):
        """异步获取会话中保存的帧评分数据"""
        try:
            session = EvalSession.objects.get(session_id=self.session_id)
            scores = dict(session.frame_scores) if session.frame_scores else {}
            if scores:  # 只有当有评分数据时才记录日志
                logger.info(f"获取会话 {self.session_id} 的帧评分数据: {scores}")
            return scores
        except EvalSession.DoesNotExist:
            logger.error(f"找不到会话: {self.session_id}")
            return {}

    
    @database_sync_to_async
    def _get_video_landmarks(self):
        """异步获取会话中保存的帧评分数据"""
        try:
            session = EvalSession.objects.get(session_id=self.session_id)
            std_frame_data = session.frame_data['std_frame_data']
            exe_frame_data = session.frame_data['exe_frame_data']
            logger.info(f"获取会话 {self.session_id} 的帧评分数据: {std_frame_data}, {exe_frame_data}")
            return std_frame_data, exe_frame_data

        except EvalSession.DoesNotExist:
            logger.error(f"找不到会话: {self.session_id}")
            return {}
    
    async def _wait_for_scores(self):
        """等待评分数据准备完成"""
        retries = 0
        while retries < self.max_retries:
            scores = await self._get_frame_scores()
            # 确保有足够的帧数据
            if scores and len(scores) > 10:  # 或者其他合适的帧数阈值
                self.frame_scores = scores
                self.scores_initialized = True
                logger.info(f"评分数据准备完成: {len(scores)} 帧")
                return True
            retries += 1
            await asyncio.sleep(self.retry_interval)
            logger.debug(f"等待评分数据... ({retries}/{self.max_retries})")
        
        logger.error("等待评分数据超时")
        return False

    async def recv(self):
        try:
            # 确保评分数据已初始化
            if not self.scores_initialized:
                success = await self._wait_for_scores()
                if not success:
                    logger.error("无法获取评分数据,停止视频播放")
                    self.stop()
                    raise RuntimeError("无法获取评分数据")

            ret, frame = self.cap.read()
            if not ret:
                # 如果是练习视频结束
                if self.type == 'exercise':
                    self.ended = True
                    self.stop()
                    # 计算并发送最终平均分
                    if self.played_frame_scores:
                        avg_score = sum(self.played_frame_scores) / len(self.played_frame_scores)
                        try:
                            await self.channel_layer.group_send(
                                f'session_{self.session_id}',
                                {
                                    'type': 'score_update',
                                    'score': avg_score,
                                    'status': 'completed',
                                    'final': True  # 标记这是最终分数
                                }
                            )
                            logger.info(f"练习视频播放完毕，发送最终平均分: {avg_score}")
                        except Exception as e:
                            logger.error(f"发送最终评分失败: {str(e)}")
                    raise RuntimeError("视频已播放完毕")
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.video_service.process_frame_for_stream(frame_rgb)
            
            # 获取当前帧的评分
            current_score = self.frame_scores.get(str(self.frame_count), 
                                                self.played_frame_scores[-1] if self.played_frame_scores else 0)
            logger.debug(f"帧 {self.frame_count} 的评分: {current_score}")
            
            # 记录当前帧的评分
            self.played_frame_scores.append(current_score)
            
            # 在帧上显示评分
            cv2.putText(
                processed_frame,
                f"Score: {current_score:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # 创建 VideoFrame
            video_frame = VideoFrame.from_ndarray(processed_frame, format="rgb24")
            pts = self.frame_count * int(1000000000 / 30)
            video_frame.pts = pts
            video_frame.time_base = fractions.Fraction(1, 1000000000)
            
            self.frame_count += 1
            return video_frame

        except Exception as e:
            logger.error(f"处理视频帧失败: {str(e)}")
            raise


    def stop(self):
        """释放资源"""
        super().stop()
        if self.cap:
            self.cap.release() 