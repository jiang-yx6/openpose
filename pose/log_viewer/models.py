from django.db import models

class LogEntry(models.Model):
    """Model to store log entries if needed for long-term storage"""
    timestamp = models.DateTimeField(auto_now_add=True)
    level = models.CharField(max_length=10)
    logger_name = models.CharField(max_length=100)
    message = models.TextField()
    exception = models.TextField(null=True, blank=True)
    
    class Meta:
        verbose_name_plural = "Log Entries"
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"[{self.timestamp}] {self.level}: {self.message[:50]}..."
