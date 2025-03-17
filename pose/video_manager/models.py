from django.db import models

class VideoAsset(models.Model):
    """Model to store video assets with their covers and tags"""
    original_mp4_path = models.CharField(max_length=500)  # Original file path
    original_cover_path = models.CharField(max_length=500)  # Original file path
    mp4_path = models.CharField(max_length=500)       # Path to the mp4 file
    cover_path = models.CharField(max_length=500)     # Path to the webp cover
    
    # Store the individual tags
    tag1 = models.CharField(max_length=100, blank=True, null=True)
    tag2 = models.CharField(max_length=100, blank=True, null=True)
    tag3 = models.CharField(max_length=100, blank=True, null=True)
    tag4 = models.CharField(max_length=100, blank=True, null=True)
    tag5 = models.CharField(max_length=100, blank=True, null=True)
    
    # Composite tag string for easy lookups
    tag_string = models.CharField(max_length=500, blank=True, null=True)
    
    # Numeric identifier (from folder/file naming)
    numeric_id = models.CharField(max_length=20, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Video: {self.original_mp4_path}, Cover: {self.original_cover_path}"