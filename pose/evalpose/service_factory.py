import logging
import os
from django.conf import settings
from .services import VideoProcessingService
from .modern_pose_analyzer import ModernPoseAnalyzer
from .pose_analyze.evaluation import detect_action_stages, select_lowest_score_frames

logger = logging.getLogger(__name__)

def get_video_processing_service(use_modern=True):
    """
    Factory function to get either the modern or legacy video processing service.
    
    Args:
        use_modern (bool): Whether to use the modern implementation
        
    Returns:
        Object: Video processing service instance
    """
    if use_modern:
        logger.info("Using modern pose analyzer implementation")
        return ModernPoseAnalyzerService()
    else:
        logger.info("Using legacy pose analyzer implementation")
        return VideoProcessingService()

class ModernPoseAnalyzerService(VideoProcessingService):
    """
    Enhanced service that integrates all new pose analyzer capabilities
    while maintaining compatibility with existing code.
    """
    
    def __init__(self):
        super().__init__()
        self.modern_analyzer = ModernPoseAnalyzer()
        logger.info("Initialized Modern Pose Analyzer Service")
        
    def process_videos(self, session_id, standard_video_path, exercise_video_path):
        """
        Process videos using the enhanced implementation.
        
        Args:
            session_id: Session identifier
            standard_video_path: Path to standard video file
            exercise_video_path: Path to exercise video file
            
        Returns:
            dict: Processing results
        """
        try:
            # Process videos with enhanced analysis
            modern_result = self.modern_analyzer.process_videos(
                session_id, 
                standard_video_path, 
                exercise_video_path
            )
            
            if not modern_result['dtw_success']:
                raise Exception(f"Modern pose analysis failed: {modern_result.get('error', 'Unknown error')}")
            
            # Update session with analysis results
            self._update_session_with_results(session_id, modern_result)
            
            # Create visualization videos
            visualization_result = self.modern_analyzer.generate_visualization_videos(
                session_id,
                modern_result['std_sequence'],
                modern_result['exe_sequence'],
                modern_result['dtw_result']
            )
            
            # Continue with HLS conversion using parent class
            result = super().process_videos(session_id, standard_video_path, exercise_video_path)
            
            # Add enhanced metrics to result
            result['advanced_metrics'] = modern_result.get('additional_metrics', {})
            result['action_stages'] = modern_result.get('action_stages', [])
            result['lowest_score_frames'] = modern_result.get('lowest_score_frames', [])
            
            return result
            
        except Exception as e:
            logger.error(f"Modern video processing failed: {str(e)}", exc_info=True)
            raise
    
    def _update_session_with_results(self, session_id, analysis_result):
        """
        Update session record with analysis results.
        """
        from .models import EvalSession
        try:
            session = EvalSession.objects.get(pk=session_id)
            session.dtw_distance = analysis_result['dtw_distance']
            session.similarity_score = analysis_result['similarity_score']
            session.frame_scores = analysis_result['frame_scores']
            
            # Store metrics with consistent key name
            advanced_metrics = analysis_result.get('additional_metrics', {})
            
            session.frame_data = {
                'std_frame_data': analysis_result['std_sequence'], 
                'exercise_frame_data': analysis_result['exe_sequence'],
                'advanced_metrics': advanced_metrics,  # Use consistent key name
                'additional_metrics': advanced_metrics  # Keep both for backward compatibility
            }
            session.status = 'completed'
            session.save()
            logger.info(f"Updated session {session_id} with modern analysis results")
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {str(e)}")
            raise
    
    # New API methods exposing enhanced functionality
    
    def get_advanced_metrics(self, session_id):
        """
        Get advanced metrics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            dict: Advanced metrics including alignment and timing analysis
        """
        from .models import EvalSession
        try:
            session = EvalSession.objects.get(pk=session_id)
            # Check both keys to be safe
            if session.frame_data:
                if 'advanced_metrics' in session.frame_data:
                    return session.frame_data['advanced_metrics']
                elif 'additional_metrics' in session.frame_data:
                    return session.frame_data['additional_metrics']
            return {}
        except Exception as e:
            logger.error(f"Error getting advanced metrics for session {session_id}: {str(e)}")
            return {}
    
    def detect_exercise_stages(self, session_id):
        """
        Detect exercise stages for a completed session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            list: Detected stages as (start_frame, end_frame) tuples
        """
        from .models import EvalSession
        try:
            session = EvalSession.objects.get(pk=session_id)
            if session.frame_data and 'exercise_frame_data' in session.frame_data:
                return detect_action_stages(session.frame_data['exercise_frame_data'])
            return []
        except Exception as e:
            logger.error(f"Error detecting exercise stages for session {session_id}: {str(e)}")
            return []
    
    def get_speed_analysis(self, session_id):
        """
        Analyze speed variation in exercise performance.
        
        Args:
            session_id: Session identifier
            
        Returns:
            dict: Speed analysis with slow and fast segments
        """
        try:
            advanced_metrics = self.get_advanced_metrics(session_id)
            if 'speed_variation' in advanced_metrics:
                return advanced_metrics['speed_variation']
            return {'slow_segments': [], 'fast_segments': []}
        except Exception as e:
            logger.error(f"Error getting speed analysis for session {session_id}: {str(e)}")
            return {'slow_segments': [], 'fast_segments': []}
    
    def get_worst_frames(self, session_id, max_frames=3):
        """
        Get frames with worst performance.
        
        Args:
            session_id: Session identifier
            max_frames: Maximum number of frames to return
            
        Returns:
            list: Worst performance frames with scores
        """
        from .models import EvalSession
        try:
            session = EvalSession.objects.get(pk=session_id)
            if not session.frame_data:
                return []
                
            stages = self.detect_exercise_stages(session_id)
            dtw_result = {'frame_scores': [(int(idx), float(score)) for idx, score in session.frame_scores.items()]}
            
            return select_lowest_score_frames(dtw_result, stages, max_frames)
        except Exception as e:
            logger.error(f"Error getting worst frames for session {session_id}: {str(e)}")
            return []