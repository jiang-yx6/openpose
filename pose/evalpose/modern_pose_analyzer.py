import os
import logging
import cv2
import numpy as np
from django.conf import settings

# Import new implementation components
from .pose_analyze.pose_detector import VideoAnalyzer
from .pose_analyze.action_comparator import ActionComparator
from .pose_analyze.evaluation import detect_action_stages, select_lowest_score_frames
from .pose_analyze.visualization import generate_video_with_selected_frames, draw_bone
from .pose_analyze.video_stretch import stretch_videos_to_same_length
from .pose_analyze.pose_comparison import dtw_compare, score_cos_sim, weight_match_l1, weight_match_l2

logger = logging.getLogger(__name__)

class ModernPoseAnalyzer:
    """
    Enhanced pose analyzer that integrates all new capabilities from the pose_analyze module
    while maintaining compatibility with the Django project structure.
    """
    
    def __init__(self):
        """Initialize the modern pose analyzer with the new implementation."""
        self.analyzer = VideoAnalyzer()
        logger.info("Initialized Modern Pose Analyzer")
    
    def process_video(self, video_path, skip_frames=2):
        """
        Process a single video to extract pose sequence data.
        
        Args:
            video_path: Path to video file
            skip_frames: Number of frames to skip (for performance)
            
        Returns:
            List of dictionaries containing landmarks and angles for each frame
        """
        try:
            logger.info(f"Processing video: {video_path}")
            return self.analyzer.process_video(video_path, skip_frames)
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise
    
    def process_videos(self, session_id, standard_video_path, exercise_video_path):
        """
        Process standard and exercise videos and perform comparison analysis.
        
        Args:
            session_id: Session identifier
            standard_video_path: Path to standard/reference video
            exercise_video_path: Path to exercise/patient video
            
        Returns:
            dict: Complete analysis results
        """
        result = {
            'dtw_success': False,
            'hls_success': False,
            'standard_hls': False,
            'exercise_hls': False,
            'overlap_hls': False,
            'frame_scores': {}
        }
        
        try:
            # Process videos
            std_sequence = self.process_video(standard_video_path)
            exe_sequence = self.process_video(exercise_video_path)
            
            # Compare sequences 
            comparator = ActionComparator(std_sequence, exe_sequence)
            dtw_result = comparator.compare_sequences()
            
            # Format frame scores to match expected format
            frame_scores = {str(idx): float(score) for idx, score in dtw_result['frame_scores']}
            result['frame_scores'] = frame_scores
            
            # Advanced metrics from new implementation
            result['additional_metrics'] = self.get_advanced_metrics(dtw_result, comparator)
            
            # Core metrics
            result['dtw_distance'] = float(dtw_result['dtw_distance'])
            result['similarity_score'] = float(dtw_result['similarity_score'] * 100)
            result['dtw_success'] = True
            
            # Action stages detection
            stages = detect_action_stages(exe_sequence)
            result['action_stages'] = stages
            
            # Get worst performance frames
            lowest_score_frames = select_lowest_score_frames(dtw_result, stages)
            result['lowest_score_frames'] = lowest_score_frames
            
            # Reference to sequences for further processing
            result['std_sequence'] = std_sequence
            result['exe_sequence'] = exe_sequence
            result['dtw_result'] = dtw_result
            
            logger.info(f"Modern pose analysis complete: similarity={result['similarity_score']:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error in modern pose analysis: {str(e)}", exc_info=True)
            return {
                'dtw_success': False,
                'error': str(e)
            }
    
    def get_advanced_metrics(self, dtw_result, comparator=None):
        """
        Calculate advanced metrics from DTW result using the new implementation.
        
        Args:
            dtw_result: DTW alignment result
            comparator: Optional ActionComparator instance
            
        Returns:
            dict: Advanced metrics
        """
        if comparator is None and 'std_sequence' in vars(self) and 'exe_sequence' in vars(self):
            comparator = ActionComparator(self.std_sequence, self.exe_sequence)
        
        if not comparator:
            logger.warning("No comparator available for advanced metrics")
            return {}
        
        try:
            alignment_path = dtw_result['alignment_path']
            return {
                'alignment_angle': comparator.compute_alignment_angle_score(alignment_path),
                'time_variance': comparator.dtw_time_variance_score(alignment_path),
                'alignment_ratio': comparator.compute_alignment_ratio(alignment_path),
                'speed_variation': comparator.analyze_speed_variation(alignment_path),
                'overall_scores': comparator.compare_overall_video()
            }
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
            return {}
    
    def generate_visualization_videos(self, session_id, std_sequence, exe_sequence, dtw_result):
        """
        Generate all visualization videos for a session using new implementation.
        
        Args:
            session_id: Session identifier
            std_sequence: Standard sequence data
            exe_sequence: Exercise sequence data  
            dtw_result: DTW comparison result
            
        Returns:
            dict: Paths to generated videos and success status
        """
        result = {
            'standard_video_success': False,
            'exercise_video_success': False, 
            'overlap_video_success': False,
            'video_paths': {}
        }
        
        try:
            # Create output directory
            output_dir = os.path.join(settings.MEDIA_ROOT, 'analysis', str(session_id))
            os.makedirs(output_dir, exist_ok=True)
            
            # Detect action stages
            stages = detect_action_stages(exe_sequence)
            
            # Generate overlap comparison video
            overlap_path = os.path.join(output_dir, 'overlap_video.mp4')
            exercise_video_path = self._get_video_path_for_sequence(session_id, 'exercise')
            
            # Use the new implementation to generate comparison video
            generate_video_with_selected_frames(
                std_sequence,
                exe_sequence,
                dtw_result,
                overlap_path,
                exercise_video_path,
                stages
            )
            
            result['overlap_video_success'] = True
            result['video_paths']['overlap'] = overlap_path
            
            logger.info(f"Generated visualization videos for session {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating visualization videos: {str(e)}")
            return result
    
    def _get_video_path_for_sequence(self, session_id, video_type):
        """
        Helper method to get video path from session.
        
        Args:
            session_id: Session identifier
            video_type: 'standard' or 'exercise'
            
        Returns:
            str: Path to video file
        """
        from .models import EvalSession, VideoFile
        try:
            session = EvalSession.objects.get(pk=session_id)
            video = VideoFile.objects.get(session=session, video_type=video_type)
            return video.file.path
        except Exception as e:
            logger.error(f"Error getting {video_type} video path: {str(e)}")
            return None
    
    def stretch_videos(self, video_path1, video_path2, output_path1, output_path2):
        """
        Stretch or compress two videos to have the same number of frames.
        New functionality from video_stretch module.
        
        Args:
            video_path1: Path to first video
            video_path2: Path to second video
            output_path1: Output path for stretched first video
            output_path2: Output path for stretched second video
            
        Returns:
            int: Number of frames in stretched videos
        """
        try:
            return stretch_videos_to_same_length(
                video_path1, video_path2, 
                output_path1, output_path2
            )
        except Exception as e:
            logger.error(f"Error stretching videos: {str(e)}")
            raise
    
    def compare_poses(self, pose1, pose2, method='dtw'):
        """
        Compare two poses using various comparison methods from pose_comparison.
        
        Args:
            pose1: First pose frame data
            pose2: Second pose frame data
            method: Comparison method ('dtw', 'cosine', 'l1', or 'l2')
            
        Returns:
            float: Similarity score (0-100)
        """
        try:
            if method == 'cosine':
                return score_cos_sim(pose1, pose2)
            elif method == 'l1':
                return weight_match_l1(pose1, pose2)
            elif method == 'l2':
                return weight_match_l2(pose1, pose2)
            else:
                # Default to DTW for sequences
                if isinstance(pose1, list) and isinstance(pose2, list):
                    _, similarity = dtw_compare(pose1, pose2)
                    return similarity
                else:
                    return score_cos_sim(pose1, pose2)
        except Exception as e:
            logger.error(f"Error comparing poses: {str(e)}")
            return 0.0