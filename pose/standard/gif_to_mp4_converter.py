import os
import subprocess
from PIL import Image
import imageio

def convert_gif_to_mp4(gif_path, output_path):
    """Convert a GIF file to MP4 using FFmpeg"""
    cmd = [
        'ffmpeg',
        '-i', gif_path,
        '-movflags', 'faststart',
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-y',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def extract_middle_frame_from_gif(gif_path, output_path):
    """Extract a frame from the middle of a GIF and save as WEBP"""
    gif_frames = imageio.mimread(gif_path, memtest=False)
    frame_count = len(gif_frames)
    
    # Select the middle frame (not the first one)
    middle_index = frame_count // 2
    if middle_index == 0 and frame_count > 1:  # If there are only 2 frames, take the second one
        middle_index = 1
    
    middle_frame = gif_frames[middle_index]
    img = Image.fromarray(middle_frame)
    img.save(output_path, 'WEBP')
    print(f"Extracted middle frame ({middle_index+1}/{frame_count}) from {gif_path} as cover")

def extract_middle_frame_from_mp4(mp4_path, output_path):
    """Extract a frame from the middle of an MP4 and save as WEBP"""
    # First get the duration of the video
    probe_cmd = [
        'ffprobe', 
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        mp4_path
    ]
    duration = float(subprocess.run(probe_cmd, check=True, capture_output=True, text=True).stdout.strip())
    
    # Extract frame from the middle of the video
    middle_time = duration / 2
    cmd = [
        'ffmpeg',
        '-ss', str(middle_time),
        '-i', mp4_path,
        '-vframes', '1',
        '-f', 'image2',
        '-y',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Extracted middle frame from {mp4_path} as cover")

def create_mp4_from_webp(webp_path, output_path, duration=10, fps=30):
    """Create an MP4 video from a WEBP image with specified duration and fps"""
    img = Image.open(webp_path)
    width, height = img.size
    
    # Ensure even dimensions for video encoding
    width = width if width % 2 == 0 else width + 1
    height = height if height % 2 == 0 else height + 1
    
    # Create a temporary file for the resized image if needed
    temp_path = webp_path.replace('.webp', '_temp.webp')
    if width != img.size[0] or height != img.size[1]:
        img_resized = img.resize((width, height))
        img_resized.save(temp_path, 'WEBP')
        input_path = temp_path
    else:
        input_path = webp_path
    
    # Create video from the image
    cmd = [
        'ffmpeg',
        '-loop', '1',
        '-i', input_path,
        '-c:v', 'libx264',
        '-t', str(duration),
        '-pix_fmt', 'yuv420p',
        '-r', str(fps),
        '-y',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Remove temporary file if created
    if os.path.exists(temp_path):
        os.remove(temp_path)

def process_directory(root_dir):
    """Process all files in the directory structure according to requirements"""
    for folder_path, _, files in os.walk(root_dir):
        # Group files by their base name (without extension)
        file_groups = {}
        
        for file in files:
            if file.endswith(('.gif', '.webp', '.mp4')):
                base_name = os.path.splitext(file)[0]
                if base_name not in file_groups:
                    file_groups[base_name] = {'gif': None, 'webp': None, 'mp4': None}
                
                if file.endswith('.gif'):
                    file_groups[base_name]['gif'] = os.path.join(folder_path, file)
                elif file.endswith('.webp'):
                    file_groups[base_name]['webp'] = os.path.join(folder_path, file)
                elif file.endswith('.mp4'):
                    file_groups[base_name]['mp4'] = os.path.join(folder_path, file)
        
        # Process each group
        for base_name, files in file_groups.items():
            gif_path = files['gif']
            webp_path = files['webp']
            mp4_path = os.path.join(folder_path, f"{base_name}.mp4") if files['mp4'] is None else files['mp4']
            
            # Case 1: Only GIF exists
            if gif_path and not webp_path:
                print(f"Converting {gif_path} to MP4 and extracting middle frame as cover")
                webp_cover_path = os.path.join(folder_path, f"{base_name}.webp")
                convert_gif_to_mp4(gif_path, mp4_path)
                extract_middle_frame_from_gif(gif_path, webp_cover_path)
            
            # Case 2: Only WEBP exists
            elif webp_path and not gif_path:
                print(f"Creating MP4 from {webp_path}")
                create_mp4_from_webp(webp_path, mp4_path)
            
            # Case 3: Both GIF and WEBP exist
            elif gif_path and webp_path:
                # if mp4 exists, skip conversion
                if os.path.exists(mp4_path):
                    print(f"MP4 already exists for {base_name}, skipping conversion")
                    continue
                print(f"Converting {gif_path} to MP4 (using existing {webp_path} as cover)")
                convert_gif_to_mp4(gif_path, mp4_path)
            
            # Case 4: only mp4 exists, just need to extract the cover
            elif os.path.exists(mp4_path) and not gif_path and not webp_path:
                print(f"Extracting middle frame from {mp4_path} as cover")
                webp_cover_path = os.path.join(folder_path, f"{base_name}.webp")
                extract_middle_frame_from_mp4(mp4_path, webp_cover_path)

if __name__ == "__main__":
    # Replace with your actual root directory
    root_directory = input("Enter the root directory path (e.g., /path/to/abc): ")
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in PATH. Please install FFmpeg first.")
        exit(1)
    
    # Process all directories
    process_directory(root_directory)
    print("Conversion completed successfully!")