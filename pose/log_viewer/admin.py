from django.contrib import admin
from django.template.response import TemplateResponse
from django.urls import path
import os
from django.conf import settings
import re
from datetime import datetime
from .models import LogEntry

@admin.register(LogEntry)
class LogEntryAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'level', 'logger_name', 'message')
    list_filter = ('level', 'logger_name', 'timestamp')
    search_fields = ('message', 'logger_name')
    readonly_fields = ('timestamp', 'level', 'logger_name', 'message', 'exception')

class LogViewerAdmin(admin.ModelAdmin):
    """Admin view for viewing log files"""
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('view-logs/', self.admin_site.admin_view(self.view_logs), name='view-logs'),
        ]
        return custom_urls + urls
        
    def view_logs(self, request):
        """View logs from log files"""
        # Get available log files
        log_dir = getattr(settings, 'LOG_DIR', os.path.join(settings.BASE_DIR, 'logs'))
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        log_files = []
        
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                if file.endswith('.log'):
                    log_files.append(file)

        # If no log files exist, create a dummy one for testing
        if not log_files:
            dummy_log_path = os.path.join(log_dir, 'django.log')
            with open(dummy_log_path, 'a') as f:
                f.write("2025-04-20 12:00:00 [INFO] django: Log viewer initialized\n")
            log_files = ['django.log']

        selected_log = request.GET.get('log_file', '')
        if not selected_log and log_files:
            selected_log = log_files[0]
        
        # Get filter parameters
        level_filter = request.GET.get('level', '')
        search_term = request.GET.get('search', '')
        
        # Default content
        log_content = []
        
        # Read the selected log file
        if selected_log and os.path.exists(os.path.join(log_dir, selected_log)):
            log_path = os.path.join(log_dir, selected_log)
            with open(log_path, 'r', encoding='utf-8', errors='replace') as file:
                lines = file.readlines()
                
                # Parse log lines
                for line in lines:
                    # Basic parsing of log format
                    level_match = re.search(r'\[(INFO|WARNING|ERROR|DEBUG|CRITICAL)\]', line)
                    level = level_match.group(1) if level_match else 'UNKNOWN'
                    
                    # Apply filtering
                    if level_filter and level != level_filter:
                        continue
                    
                    if search_term and search_term.lower() not in line.lower():
                        continue
                    
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    timestamp = timestamp_match.group(1) if timestamp_match else ""
                    
                    log_content.append({
                        'timestamp': timestamp,
                        'level': level,
                        'message': line.strip()
                    })
            
            # Sort by timestamp (newest first)
            log_content.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Paginate results
        page = int(request.GET.get('page', 1))
        per_page = 100
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        total_pages = (len(log_content) + per_page - 1) // per_page
        paginated_logs = log_content[start_idx:end_idx]
        
        context = {
            'log_files': log_files,
            'selected_log': selected_log,
            'log_content': paginated_logs,
            'level_filter': level_filter,
            'search_term': search_term,
            'levels': ['INFO', 'WARNING', 'ERROR', 'DEBUG', 'CRITICAL'],
            'page': page,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'next_page': page + 1,
            'prev_page': page - 1,
            'title': 'Log Viewer',
            'app_label': 'log_viewer',
            'opts': {'app_label': 'log_viewer'},
        }
        
        return TemplateResponse(request, 'admin/log_viewer/view_logs.html', context)

# Register with a proxy model to avoid the database issue
class LogViewer(LogEntry):
    class Meta:
        proxy = True
        verbose_name = "Log Viewer"
        verbose_name_plural = "Log Viewer"

admin.site.register(LogViewer, LogViewerAdmin)
