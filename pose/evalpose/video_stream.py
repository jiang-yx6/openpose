import cv2
from av import VideoFrame
from aiortc import MediaStreamTrack
import fractions
import logging

logger = logging.getLogger(__name__)

class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, video_path, video_service):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_service = video_service
        self.frame_count = 0
        self.ended = False

    async def recv(self):
        try:
            if self.ended:
                raise RuntimeError("视频已播放完毕")

            ret, frame = self.cap.read()
            if not ret:
                self.ended = True
                self.stop()
                raise RuntimeError("视频已播放完毕")

            # 转换颜色空间从 BGR 到 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 使用服务处理帧（如果需要）
            if hasattr(self.video_service, 'process_frame_for_stream'):
                frame_rgb = self.video_service.process_frame_for_stream(frame_rgb)
            
            # 创建 VideoFrame
            video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            pts = self.frame_count * int(1000000000 / 30)  # 30 FPS
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