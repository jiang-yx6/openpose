from rest_framework import serializers
from .models import EvalSession, VideoFile
import logging

logger = logging.getLogger(__name__)

class VideoUploadSerializer(serializers.Serializer):
    standardVideo = serializers.FileField()
    exerciseVideo = serializers.FileField()

    def validate(self, data):
        """
        验证上传的视频文件
        """
        logger.info("开始验证视频文件")
        allowed_types = ['video/mp4', 'video/avi', 'video/mpeg']
        max_size = 100 * 1024 * 1024  # 100MB

        for field in ['standardVideo', 'exerciseVideo']:
            video = data.get(field)
            if video:
                logger.debug(f"验证 {field}: 大小={video.size}, 类型={video.content_type}")
                if not video.content_type in allowed_types:
                    logger.warning(f"{field}: 不支持的视频格式 {video.content_type}")
                    raise serializers.ValidationError(f"{field}: 不支持的视频格式")
                if video.size > max_size:
                    logger.warning(f"{field}: 文件过大 {video.size}")
                    raise serializers.ValidationError(f"{field}: 文件大小不能超过100MB")
        
        logger.info("视频文件验证通过")
        return data

class SessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvalSession
        fields = ['session_id', 'status', 'score'] 