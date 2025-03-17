import os
from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Create directories needed for standardized video assets'

    def handle(self, *args, **options):
        # Define directories to create
        directories = [
            os.path.join(settings.BASE_DIR, 'standard', 'video'),
            os.path.join(settings.BASE_DIR, 'standard', 'cover'),
        ]
        
        # Create directories
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.stdout.write(self.style.SUCCESS(f'Created directory: {directory}'))