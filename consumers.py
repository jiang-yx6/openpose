import json
from channels.generic.websocket import AsyncWebsocketConsumer
import logging
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate
)
from .video_stream import VideoStreamTrack
from channels.db import database_sync_to_async
from .models import VideoFile
logger = logging.getLogger(__name__)

class VideoAnalysisConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """建立WebSocket连接"""
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.room_group_name = f'session_{self.session_id}'
        self.pc = None

        try:
            # 加入房间组
            await self.channel_layer.group_add(
                self.room_group_name,
                self.channel_name
            )
            await self.accept()
            
            # 设置视频流
            await self.setup_video_stream()
            
            logger.info(f"WebSocket连接已建立: session_id={self.session_id}")
        except Exception as e:
            logger.error(f"WebSocket连接建立失败: {str(e)}", exc_info=True)
            raise

    async def setup_video_stream(self):
        """设置WebRTC视频流"""
        try:
            video = await self.get_video_file()
            
            # 创建新的PeerConnection
            if self.pc:
                await self.pc.close()
            
            self.pc = RTCPeerConnection()

            # 添加视频轨道
            video_track = VideoStreamTrack(video.file.path)
            self.pc.addTrack(video_track)

            # 创建offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            # 发送offer给客户端
            await self.send(text_data=json.dumps({
                'type': 'offer',
                'offer': {
                    'sdp': self.pc.localDescription.sdp,
                    'type': self.pc.localDescription.type
                }
            }))

        except Exception as e:
            logger.error(f"设置视频流失败: {str(e)}", exc_info=True)
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def receive(self, text_data):
        """接收前端消息"""
        try:
            data = json.loads(text_data)
            
            if data['type'] == 'answer':
                if self.pc:
                    answer = RTCSessionDescription(
                        sdp=data['answer']['sdp'],
                        type=data['answer']['type']
                    )
                    await self.pc.setRemoteDescription(answer)

            elif data['type'] == 'ice-candidate':
                if self.pc:
                    candidate = RTCIceCandidate(
                        sdpMid=data['candidate']['sdpMid'],
                        sdpMLineIndex=data['candidate']['sdpMLineIndex'],
                        candidate=data['candidate']['candidate']
                    )
                    await self.pc.addIceCandidate(candidate)

        except Exception as e:
            logger.error(f"处理消息时发生错误: {str(e)}", exc_info=True)
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def score_update(self, event):
        """处理并发送评分更新"""
        try:
            await self.send(text_data=json.dumps({
                'type': 'score_update',
                'score': event['score'],
                'status': event['status']
            }))
        except Exception as e:
            logger.error(f"发送评分更新失败: {str(e)}", exc_info=True)

    async def disconnect(self, close_code):
        """断开WebSocket连接"""
        try:
            if self.pc:
                await self.pc.close()
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )
        except Exception as e:
            logger.error(f"断开连接时发生错误: {str(e)}")

    @database_sync_to_async
    def get_video_file(self):
        """
        获取视频文件（同步操作）
        """
        return VideoFile.objects.get(
            session__session_id=self.session_id,
            video_type='standard'
        )