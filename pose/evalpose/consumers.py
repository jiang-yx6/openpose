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
from .services import VideoProcessingService
import asyncio

logger = logging.getLogger(__name__)

class VideoAnalysisConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """建立WebSocket连接"""
        try:
            # 从URL参数获取session_id和type
            self.session_id = self.scope['url_route']['kwargs']['session_id']
            self.type = self.scope['url_route']['kwargs']['type']
            self.room_group_name = f'session_{self.session_id}'
            self.pc = None

            # 加入房间组
            await self.channel_layer.group_add(
                self.room_group_name,
                self.channel_name
            )
            
            # 先接受连接
            await self.accept()
            
            logger.info(f"WebSocket连接已建立: session_id={self.session_id}, type={self.type}")

            # 如果是标准视频，设置WebRTC
            await self.setup_video_stream()
            # 如果是练习视频连接，开始视频分析
            asyncio.create_task(self.start_video_analysis())
                
        except Exception as e:
            logger.error(f"WebSocket连接建立失败: {str(e)}", exc_info=True)
            raise

    async def setup_video_stream(self):
        """设置WebRTC视频流"""
        try:
            video = await self.get_video_file()
            if not video:
                logger.error(f"找不到视频文件: session_id={self.session_id}, type={self.type}")
                return
                
            self.pc = RTCPeerConnection()
            video_track = VideoStreamTrack(video.file.path)
            self.pc.addTrack(video_track)
            
            # 创建offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            await self.send(text_data=json.dumps({
                'type': 'offer',
                'offer': {
                    'sdp': self.pc.localDescription.sdp,
                    'type': self.pc.localDescription.type
                }
            }))
            logger.info(f"已发送WebRTC offer: session_id={self.session_id}")
            
        except Exception as e:
            logger.error(f"设置视频流失败: {str(e)}", exc_info=True)
            raise

    async def start_video_analysis(self):
        """开始视频分析"""
        try:
            # 获取视频文件
            standard_video = await self.get_video_file('standard')
            exercise_video = await self.get_video_file('exercise')
            
            if not standard_video or not exercise_video:
                logger.error(f"找不到视频文件: session_id={self.session_id}")
                return
                
            # 开始处理视频并发送实时反馈
            await self.process_and_stream_video(
                standard_video.file.path,
                exercise_video.file.path
            )
            
        except Exception as e:
            logger.error(f"视频分析失败: {str(e)}", exc_info=True)
            raise

    @database_sync_to_async
    def get_video_file(self, video_type=None):
        """获取视频文件"""
        try:
            return VideoFile.objects.get(
                session__session_id=self.session_id,
                video_type=video_type if video_type else self.type
            )
        except VideoFile.DoesNotExist:
            logger.error(f"找不到视频文件: session_id={self.session_id}, type={video_type if video_type else self.type}")
            return None

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

    async def process_and_stream_video(self, standard_video_path, exercise_video_path):
        """处理视频并发送评分"""
        try:
            # 创建视频处理服务实例
            video_service = VideoProcessingService()
            
            # 调用处理方法
            await database_sync_to_async(video_service.process_videos)(
                self.session_id,
                standard_video_path,
                exercise_video_path
            )
            
            logger.info(f"视频处理完成: session_id={self.session_id}")
            
        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}", exc_info=True)
            # 发送错误消息给前端
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))