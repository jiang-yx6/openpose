from django.contrib import admin
from .models import EvalSession, VideoFile, VideoConfig

class VideoConfigAdmin(admin.ModelAdmin):
    list_display = ('numeric_id', 'description', 'updated_at')
    search_fields = ('numeric_id', 'description')

admin.site.register(VideoConfig, VideoConfigAdmin)

# Register other models if not already registered
admin.site.register(EvalSession)
admin.site.register(VideoFile)
