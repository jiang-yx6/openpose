import os
import logging
from django.conf import settings
import numpy as np

# Import new implementation components
from .pose_analyze.pose_detector import VideoAnalyzer as NewVideoAnalyzer
from .pose_analyze.action_comparator import ActionComparator as NewActionComparator
from .pose_analyze.evaluation import detect_action_stages, select_lowest_score_frames

logger = logging.getLogger(__name__)

class ModernPoseAnalyzer:
    """
    Adapter class that integrates the new pose analyzer implementation
    with the existing Django project without modifying existing code.
    """
    
    def __init__(self):
        """Initialize the modern pose analyzer with the new implementation."""
        self.analyzer = NewVideoAnalyzer()
        logger.info("Initialized Modern Pose Analyzer")
        
    def process_videos(self, session_id, standard_video_path, exercise_video_path):
        """
        Process videos using the new implementation and return results compatible
        with the existing system's expectations.
        
        Args:
            session_id: Session ID for organizing outputs
            standard_video_path: Path to standard video file
            exercise_video_path: Path to exercise video file
            
        Returns:
            dict: Result data including DTW distance, similarity score, frame scores, etc.
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
            # Process videos to extract pose data using new implementation
            logger.info(f"Processing standard video: {standard_video_path}")
            std_sequence = self.analyzer.process_video(standard_video_path)
            
            logger.info(f"Processing exercise video: {exercise_video_path}")
            exe_sequence = self.analyzer.process_video(exercise_video_path)
            
            # Compare sequences using new implementation
            logger.info("Comparing sequences with new implementation")
            comparator = NewActionComparator(std_sequence, exe_sequence)
            dtw_result = comparator.compare_sequences()
            
            # Extract and format frame scores to match expected format
            frame_scores = {str(idx): float(score) for idx, score in dtw_result['frame_scores']}
            result['frame_scores'] = frame_scores
            
            # Collect additional metrics from the new implementation
            additional_metrics = {
                'alignment_angle': comparator.compute_alignment_angle_score(dtw_result['alignment_path']),
                'time_variance': comparator.dtw_time_variance_score(dtw_result['alignment_path']),
                'alignment_ratio': comparator.compute_alignment_ratio(dtw_result['alignment_path']),
                'speed_variation': comparator.analyze_speed_variation(dtw_result['alignment_path']),
                'overall_scores': comparator.compare_overall_video()
            }
            
            # Add core metrics to result
            result['dtw_distance'] = float(dtw_result['dtw_distance'])
            result['similarity_score'] = float(dtw_result['similarity_score'] * 100)
            result['dtw_success'] = True
            result['additional_metrics'] = additional_metrics
            
            # Return sequences for further processing
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
    
    def process_overlap_video(self, std_sequence, exe_sequence, dtw_result, output_path, exercise_video_path):
        """
        Generate overlap video using the new implementation.
        
        Args:
            std_sequence: Standard sequence data
            exe_sequence: Exercise sequence data
            dtw_result: DTW comparison result
            output_path: Path to save output video
            exercise_video_path: Path to exercise video file
            
        Returns:
            bool: Success status
        """
        try:
            # Detect action stages for better visualization
            stages = detect_action_stages(exe_sequence)
            
            # Generate visualization using the new implementation's _process_overlap_video method
            self.analyzer._process_overlap_video(
                std_sequence,
                exe_sequence,
                dtw_result,
                output_path,
                exercise_video_path,
                save_lowest_scores=True
            )
            return True
        except Exception as e:
            logger.error(f"Error generating overlap video: {str(e)}", exc_info=True)
            return False