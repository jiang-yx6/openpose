import re
from pathlib import Path
from django.core.management.base import BaseCommand
from video_manager.models import VideoAsset
from django.conf import settings

class Command(BaseCommand):
    help = 'Import videos and covers into the database'

    def add_arguments(self, parser):
        parser.add_argument('--path', type=str, default='static/example_videos',
                           help='Path to the videos directory')
        parser.add_argument('--clear', action='store_true',
                           help='Clear existing database entries')

    def handle(self, *args, **options):
        if options['clear']:
            self.stdout.write('Clearing existing database entries...')
            VideoAsset.objects.all().delete()

        videos_path = Path(settings.BASE_DIR) / options['path']
        self.import_videos(videos_path)
        self.stdout.write(self.style.SUCCESS('Successfully imported videos'))

    def extract_file_id_and_tags(self, filename):
        """Extract file ID and tags from filename like 01_tag1.1_tag1.1.1_tag1.1.1.1"""
        # Remove extension
        base_name = filename.stem
        
        # Split by underscore
        parts = base_name.split('_')
        
        # First part is file ID, rest are file tags
        file_id = parts[0]
        file_tags = parts[1:] if len(parts) > 1 else []
        
        return file_id, file_tags

    def extract_folder_id_and_tag(self, folder_name):
        """Extract folder ID and tag from folder name like 01_tag1"""
        parts = folder_name.split('_', 1)
        
        if len(parts) > 1:
            folder_id = parts[0]
            folder_tag = parts[1]
        else:
            folder_id = folder_name
            folder_tag = None
            
        return folder_id, folder_tag
    
    def to_unix_path(self, path):
        """Convert a path to Unix format (forward slashes)"""
        return str(path).replace('\\', '/')

    def import_videos(self, root_path):
        """Scan directory structure and import videos with their covers"""
        count = 0
        
        for folder_path in root_path.glob('**'):
            if not folder_path.is_dir():
                continue
                
            # Extract folder info
            folder_name = folder_path.name
            folder_id, folder_tag = self.extract_folder_id_and_tag(folder_name)
            
            mp4_files = {f for f in folder_path.glob('*.mp4')}
            webp_files = {f.name for f in folder_path.glob('*.webp')}
            
            for mp4_file in mp4_files:
                base_name = mp4_file.stem
                webp_file = f"{base_name}.webp"
                
                if webp_file in webp_files:
                    # Get relative paths
                    rel_folder = folder_path.relative_to(settings.BASE_DIR)
                    mp4_rel_path = rel_folder / mp4_file.name
                    webp_rel_path = rel_folder / webp_file
                    
                    # Convert to Unix-like paths
                    mp4_rel_path_unix = self.to_unix_path(mp4_rel_path)
                    webp_rel_path_unix = self.to_unix_path(webp_rel_path)
                    
                    # Extract file ID and tags
                    file_id, file_tags = self.extract_file_id_and_tags(mp4_file)
                    
                    # Create combined ID in format "folder_id_file_id"
                    combined_id = f"{folder_id}_{file_id}"
                    
                    # Combine tags: folder tag first, then file tags
                    all_tags = [folder_tag] + file_tags if folder_tag else file_tags
                    
                    # Ensure we have at most 5 tags
                    all_tags = all_tags[:5]
                    
                    # Pad with None if needed
                    while len(all_tags) < 5:
                        all_tags.append(None)
                    
                    # Create tag string (e.g., "tag1_tag1.1_tag1.1.2")
                    tag_string = "_".join([t for t in all_tags if t])
                    
                    # Create standard paths with Unix format
                    standard_mp4_path = f"standard/video/tag/{tag_string}.mp4"
                    standard_webp_path = f"standard/cover/tag/{tag_string}.webp"
                    
                    # Create or update database entry
                    VideoAsset.objects.create(
                        original_mp4_path=mp4_rel_path_unix,
                        original_cover_path=webp_rel_path_unix,
                        mp4_path=standard_mp4_path,
                        cover_path=standard_webp_path,
                        tag1=all_tags[0],
                        tag2=all_tags[1],
                        tag3=all_tags[2],
                        tag4=all_tags[3],
                        tag5=all_tags[4],
                        tag_string=tag_string,
                        numeric_id=combined_id
                    )
                    count += 1
                    
                    self.stdout.write(f"Imported: {mp4_rel_path_unix} with ID: {combined_id}, tags: {tag_string}")
                else:
                    self.stdout.write(self.style.WARNING(f"No cover found for {mp4_file.name}"))
                    
        self.stdout.write(f"Total videos imported: {count}")