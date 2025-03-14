from rest_framework import serializers
from .models import EvalSession, VideoFile
import logging
import base64
import uuid
import os
from django.core.files.base import ContentFile
from django.conf import settings

logger = logging.getLogger(__name__)

class VideoUploadSerializer(serializers.Serializer):
    standard = serializers.CharField()  # 接收标准视频的base64字符串
    exercise = serializers.CharField()  # 接收练习视频的base64字符串
    standardVideoName = serializers.CharField(required=False, default="standard.mp4")
    exerciseVideoName = serializers.CharField(required=False, default="exercise.mp4")

    def validate(self, data):
        """
        验证上传的视频base64数据
        """
        logger.info("开始验证base64编码的视频数据")
        max_size = 100 * 1024 * 1024  # 100MB

        for field in ['standard', 'exercise']:
            video_base64 = data.get(field)
            if video_base64:
                try:
                    # 直接解码base64数据检查大小
                    file_data = base64.b64decode(video_base64)
                    file_size = len(file_data)
                    
                    if file_size > max_size:
                        logger.warning(f"{field}: 文件过大 {file_size}")
                        raise serializers.ValidationError(f"{field}: 文件大小不能超过100MB")
                    
                except Exception as e:
                    logger.error(f"处理base64数据失败: {str(e)}")
                    raise serializers.ValidationError(f"{field}: base64数据处理失败")
        
        logger.info("视频数据验证通过")
        return data

    def base64_to_file(self, base64_data, filename):
        """将base64数据转换为文件"""
        try:
            file_data = base64.b64decode(base64_data)
            return ContentFile(file_data, name=filename)
        except Exception as e:
            logger.error(f"base64转文件失败: {str(e)}")
            raise serializers.ValidationError("base64数据转换为文件失败")

class SessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvalSession
        fields = ['session_id', 'status', 'score']