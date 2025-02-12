import cv2
import numpy as np
from av import VideoFrame
from aiortc import MediaStreamTrack
import asyncio
import fractions

class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self._start = None
        self._timestamp = 0

    async def next_timestamp(self):
        if self._start is None:
            self._start = asyncio.get_event_loop().time()
            self._timestamp = 0
        else:
            self._timestamp += int(1000000000 / 30)  # 30 FPS
        return self._timestamp

    async def recv(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.video_path}")

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Could not read video frame")

        # 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建 VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = await self.next_timestamp()
        video_frame.time_base = fractions.Fraction(1, 1000000000)
        
        return video_frame

    def stop(self):
        super().stop()
        if self.cap:
            self.cap.release() 