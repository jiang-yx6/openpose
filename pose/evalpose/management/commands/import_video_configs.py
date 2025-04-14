import json
import os
from django.core.management.base import BaseCommand
from evalpose.models import VideoConfig
from django.conf import settings

class Command(BaseCommand):
    help = 'Import video configurations from JSON file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            default=os.path.join(settings.BASE_DIR,'..', '..', '..', 'Config', 'Config.json'),
            help='Path to the JSON configuration file'
        )

    def handle(self, *args, **options):
        file_path = options['file']
        
        if not os.path.exists(file_path):
            self.stderr.write(self.style.ERROR(f'File not found: {file_path}'))
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            
            count = 0
            for filename, config in configs.items():
                # Extract the numeric part from the filename (e.g., "Actions\1.mp4" -> "01")
                video_number = filename.split('\\')[-1].split('.')[0]
                # Format the numeric_id as "01_XX" where XX is the video number padded to 2 digits
                numeric_id = f"01_{int(video_number):02d}"
                
                # Create or update the configuration
                obj, created = VideoConfig.objects.update_or_create(
                    numeric_id=numeric_id,
                    defaults={
                        'key_angles': config.get('KEY_ANGLES', {}),
                        'normalization_joints': config.get('NORMALIZATION_JOINTS', []),
                        'description': config.get('Describe', '')
                    }
                )
                
                action = "Created" if created else "Updated"
                self.stdout.write(self.style.SUCCESS(f"{action} config for {numeric_id}: {obj.description}"))
                count += 1
            
            self.stdout.write(self.style.SUCCESS(f"Successfully imported {count} configurations"))
            
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'Error importing configurations: {str(e)}'))