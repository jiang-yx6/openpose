from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required

@staff_member_required
def log_viewer(request):
    """View for displaying logs (if needed outside admin)"""
    return render(request, 'log_viewer/log_viewer.html')