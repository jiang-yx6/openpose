from django.db import models
import uuid

class EvalSession(models.Model):
    """评估会话模型"""
    STATUS_CHOICES = [
        ('pending', '等待处理'),
        ('processing', '处理中'),
        ('completed', '已完成'),
        ('failed', '失败'),
        ('cancelled', '已取消'),
    ]

    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    score = models.FloatField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    dtw_distance = models.FloatField(null=True, blank=True)
    similarity_score = models.FloatField(null=True, blank=True)
    frame_scores = models.JSONField(null=True, blank=True)
    report_path = models.CharField(max_length=255, null=True, blank=True)    
    frame_data = models.JSONField(null=True,blank=True)
    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Session {self.session_id} - {self.status}"

class VideoFile(models.Model):
    """视频文件模型"""
    VIDEO_TYPE_CHOICES = [
        ('standard', '标准视频'),
        ('exercise', '练习视频')
    ]

    session = models.ForeignKey(EvalSession, on_delete=models.CASCADE, related_name='videos')
    file = models.FileField(upload_to='videos/')
    video_type = models.CharField(max_length=20, choices=VIDEO_TYPE_CHOICES)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['session', 'video_type']

    def __str__(self):
        return f"{self.video_type} - {self.session.session_id}"

