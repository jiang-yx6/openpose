import cv2
from av import VideoFrame
from aiortc import MediaStreamTrack
import fractions
import logging

logger = logging.getLogger(__name__)

class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0

    async def recv(self):
        """读取视频帧，转换为灰度并返回"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                # 如果视频结束，重新开始
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Could not read video frame")

            # 转换为灰度图
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 转换回 RGB 格式（WebRTC需要），但实际显示的是灰度图
            frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            
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