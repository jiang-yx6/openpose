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
from .pose_analyze.config_service import get_config_class, get_config_instance
from .models import VideoConfig
from .exceptions import PoseAnalysisError, FullBodyNotVisibleError, VideoLengthMismatchError, ErrorCodes

logger = logging.getLogger(__name__)

class ModernPoseAnalyzer:
    """
    Enhanced pose analyzer that integrates all new capabilities from the pose_analyze module
    while maintaining compatibility with the Django project structure.
    Uses dynamic configuration based on numeric_id.
    """
    
    def __init__(self, numeric_id=None, config=None):
        """
        Initialize the modern pose analyzer with the new implementation.
        
        Args:
            numeric_id (str, optional): The numeric ID to load configuration from database
            config (object, optional): Direct config object if already available
        """
        if config:
            self.config = config
        elif numeric_id:
            self.config = get_config_class(numeric_id)
            logger.info(f"Loaded configuration for {numeric_id}")
        else:
            # Use default config
            from .pose_analyze.config import Config
            self.config = Config
            logger.info("Using default configuration")
            
        # Initialize analyzer with the selected config
        self.analyzer = VideoAnalyzer(config=self.config)
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
    
    def process_videos(self, session_id, standard_video_path, exercise_video_path, standard_numeric_id=None, config=None):
        """
        Process standard and exercise videos and perform comparison analysis.
        
        Args:
            session_id: Session identifier
            standard_video_path: Path to standard/reference video
            exercise_video_path: Path to exercise/patient video
            standard_numeric_id: Numeric ID of standard video for configuration
            config: Direct configuration object to use
            
        Returns:
            dict: Complete analysis results
            
        Raises:
            FullBodyNotVisibleError: When full body is not visible or pose data is inconsistent
            VideoLengthMismatchError: When there's an issue with video length or alignment
            PoseAnalysisError: For other pose analysis specific errors
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
            # Update configuration if needed
            if config and not self.config == config:
                self.config = config
                self.analyzer = VideoAnalyzer(config=self.config)
                logger.info(f"Updated analyzer with provided configuration")
            elif standard_numeric_id and not hasattr(self.config, 'NUMERIC_ID'):
                self.config = get_config_class(standard_numeric_id)
                self.analyzer = VideoAnalyzer(config=self.config)
                logger.info(f"Updated configuration for {standard_numeric_id}")
                
            # Process videos
            std_sequence = self.process_video(standard_video_path)
            exe_sequence = self.process_video(exercise_video_path)
            
            # Compare sequences 
            try:
                comparator = ActionComparator(std_sequence, exe_sequence)
                dtw_result = comparator.compare_sequences()
            except ValueError as e:
                if 'X has 4 features, but StandardScaler is expecting 70 features as input' in str(e):
                    logger.error("Mismatch in feature dimensions during DTW comparison")
                    raise FullBodyNotVisibleError()
                elif 'setting an array element with a sequence' in str(e) or 'inhomogeneous shape' in str(e):
                    logger.error("Inhomogeneous shape detected in input sequences")
                    raise FullBodyNotVisibleError()
                else:
                    raise PoseAnalysisError(f"DTW comparison error: {str(e)}", 
                                          code=400, 
                                          ui_error_code=ErrorCodes.PROCESSING_ERROR)
            except IndexError as e:
                if "list index out of range" in str(e):
                    logger.error("Index error in DTW comparison, likely due to video length issues")
                    raise VideoLengthMismatchError()
                else:
                    raise

            # Format frame scores to match expected format
            frame_scores = {str(idx): float(score) for idx, score in dtw_result['frame_scores']}
            result['frame_scores'] = frame_scores
            
            # Advanced metrics from new implementation
            result['additional_metrics'] = self.get_advanced_metrics(dtw_result, comparator)
            
            # Core metrics
            result['dtw_distance'] = float(dtw_result['dtw_distance'])
            result['similarity_score'] = float(dtw_result['similarity_score'] * 100)
            result['dtw_success'] = True
            
            # Action stages detection using the new implementation from pose_detector.py
            stages = detect_action_stages(exe_sequence)
            result['action_stages'] = stages
            
            # Get worst performance frames using the new implementation
            lowest_score_frames = select_lowest_score_frames(dtw_result, stages)
            result['lowest_score_frames'] = lowest_score_frames
            
            # Reference to sequences for further processing
            result['std_sequence'] = std_sequence
            result['exe_sequence'] = exe_sequence
            result['dtw_result'] = dtw_result
            
            # Include configuration information in the result
            if hasattr(self.config, 'NUMERIC_ID') and hasattr(self.config, 'DESCRIPTION'):
                result['config_info'] = {
                    'numeric_id': self.config.NUMERIC_ID,
                    'description': self.config.DESCRIPTION
                }
            
            logger.info(f"Modern pose analysis complete: similarity={result['similarity_score']:.2f}%")
            return result
            
        except PoseAnalysisError as e:
            # Re-raise the exception to be handled by the API error handler
            raise  
        except Exception as e:
            logger.error(f"Error in modern pose analysis: {str(e)}", exc_info=True)
            raise PoseAnalysisError(
                f"Unexpected error during pose analysis: {str(e)}", 
                code=500, 
                ui_error_code=ErrorCodes.UNKNOWN_ERROR
            )
    
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
    
    def generate_visualization_videos(self, session_id, std_sequence, exe_sequence, dtw_result, standard_numeric_id=None, config=None):
        """
        Generate all visualization videos for a session using new implementation.
        
        Args:
            session_id: Session identifier
            std_sequence: Standard sequence data
            exe_sequence: Exercise sequence data  
            dtw_result: DTW comparison result
            standard_numeric_id: Optional numeric_id to get configuration
            config: Direct configuration object to use
            
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
            # Determine which configuration to use
            if config and not self.config == config:
                use_config = config
                logger.info("Using provided configuration for visualization")
            elif standard_numeric_id and (not hasattr(self.config, 'NUMERIC_ID') or 
                                         self.config.NUMERIC_ID != standard_numeric_id):
                use_config = get_config_class(standard_numeric_id)
                logger.info(f"Using configuration from numeric_id {standard_numeric_id} for visualization")
            else:
                use_config = self.config
                logger.info("Using analyzer's current configuration for visualization")
                
            # Create output directory
            output_dir = os.path.join(settings.MEDIA_ROOT, 'analysis', str(session_id))
            os.makedirs(output_dir, exist_ok=True)
            
            # Detect action stages using the function from pose_detector.py
            stages = detect_action_stages(exe_sequence)
            
            # Generate overlap comparison video
            overlap_path = os.path.join(output_dir, 'overlap_video.mp4')
            exercise_video_path = self._get_video_path_for_sequence(session_id, 'exercise')
            
            # Use the enhanced implementation from pose_detector.py to generate comparison video with config
            self.analyzer.generate_video_with_selected_frames(
                std_sequence,
                exe_sequence,
                dtw_result,
                overlap_path,
                exercise_video_path,
                stages,
                config=use_config  # Pass the config here
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
        Compare two poses using various comparison methods.
        
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
    
    def detect_action_stages(self, sequence, angle_threshold=5):
        """
        Detect action stages in the given sequence using the implementation from pose_detector.py.
        
        Args:
            sequence: Sequence of pose frames
            angle_threshold: Threshold for angle changes to detect stage boundaries
            
        Returns:
            list: List of (start_frame, end_frame) tuples representing stages
        """
        return detect_action_stages(sequence, angle_threshold)
    
    def select_lowest_score_frames(self, dtw_result, stages, max_frames=3):
        """
        Select frames with the lowest scores from each detected stage.
        
        Args:
            dtw_result: Result from DTW comparison
            stages: List of (start_frame, end_frame) tuples representing stages
            max_frames: Maximum number of frames to return
            
        Returns:
            list: List of (frame_idx, score) tuples for lowest scoring frames
        """
        return select_lowest_score_frames(dtw_result, stages, max_frames)
    
    @staticmethod
    def get_config_from_numeric_id(numeric_id):
        """
        Static helper method to get configuration class from numeric_id
        
        Args:
            numeric_id: The standard video numeric ID (e.g. "01_01")
            
        Returns:
            Config class with dynamic configuration
        """
        return get_config_class(numeric_id)