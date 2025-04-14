import logging
import os
from django.conf import settings
from .services import VideoProcessingService
from .modern_pose_analyzer import ModernPoseAnalyzer
from .pose_analyze.evaluation import detect_action_stages, select_lowest_score_frames
from .pose_analyze.config_service import get_config_class

logger = logging.getLogger(__name__)

def get_video_processing_service(use_modern=True, numeric_id=None, config=None):
    """
    Factory function to get either the modern or legacy video processing service.
    
    Args:
        use_modern (bool): Whether to use the modern implementation
        numeric_id (str, optional): Numeric ID to configure the service with
        config (object, optional): Direct configuration object to use
        
    Returns:
        Object: Video processing service instance
    """
    if use_modern:
        if config:
            logger.info(f"Using modern pose analyzer implementation with provided config")
            return ModernPoseAnalyzerService(config=config)
        elif numeric_id:
            logger.info(f"Using modern pose analyzer implementation with config for {numeric_id}")
            return ModernPoseAnalyzerService(numeric_id=numeric_id)
        else:
            logger.info("Using modern pose analyzer implementation with default config")
            return ModernPoseAnalyzerService()
    else:
        logger.info("Using legacy pose analyzer implementation")
        return VideoProcessingService()

class ModernPoseAnalyzerService(VideoProcessingService):
    """
    Enhanced service that integrates all new pose analyzer capabilities
    while maintaining compatibility with existing code.
    """
    
    def __init__(self, numeric_id=None, config=None):
        super().__init__()
        # Initialize with dynamic configuration
        if config:
            self.modern_analyzer = ModernPoseAnalyzer(config=config)
            logger.info(f"Initialized Modern Pose Analyzer Service with provided configuration")
        elif numeric_id:
            config_class = get_config_class(numeric_id)
            self.modern_analyzer = ModernPoseAnalyzer(numeric_id=numeric_id, config=config_class)
            logger.info(f"Initialized Modern Pose Analyzer Service with configuration for {numeric_id}")
        else:
            self.modern_analyzer = ModernPoseAnalyzer()
            logger.info("Initialized Modern Pose Analyzer Service with default configuration")
        
    def process_videos(self, session_id, standard_video_path, exercise_video_path, standard_numeric_id=None, config=None):
        """
        Process videos using the enhanced implementation.
        
        Args:
            session_id: Session identifier
            standard_video_path: Path to standard video file
            exercise_video_path: Path to exercise video file
            standard_numeric_id: Optional numeric ID for standard video configuration
            config: Optional configuration object
            
        Returns:
            dict: Processing results
        """
        try:
            # If config is provided and analyzer wasn't initialized with it,
            # update the analyzer with the new config
            if config and not hasattr(self.modern_analyzer.config, 'KEY_ANGLES'):
                logger.info(f"Updating analyzer with provided configuration")
                self.modern_analyzer = ModernPoseAnalyzer(config=config)
            # If numeric_id is provided and analyzer wasn't initialized with it,
            # update configuration
            elif standard_numeric_id and not hasattr(self.modern_analyzer.config, 'NUMERIC_ID'):
                logger.info(f"Updating analyzer configuration to {standard_numeric_id}")
                self.modern_analyzer = ModernPoseAnalyzer(numeric_id=standard_numeric_id)
                
            # Process videos with enhanced analysis
            modern_result = self.modern_analyzer.process_videos(
                session_id, 
                standard_video_path, 
                exercise_video_path,
                standard_numeric_id=standard_numeric_id,
                config=config
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
                modern_result['dtw_result'],
                standard_numeric_id=standard_numeric_id,
                config=config
            )
            
            # Continue with HLS conversion using parent class
            result = super().process_videos(session_id, standard_video_path, exercise_video_path,config=config)
            
            # Add enhanced metrics to result
            result['advanced_metrics'] = modern_result.get('additional_metrics', {})
            result['action_stages'] = modern_result.get('action_stages', [])
            result['lowest_score_frames'] = modern_result.get('lowest_score_frames', [])
            
            # Include configuration info if available
            if 'config_info' in modern_result:
                result['config_info'] = modern_result['config_info']
            
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
            
            # Create frame data object
            frame_data = {
                'std_frame_data': analysis_result['std_sequence'], 
                'exercise_frame_data': analysis_result['exe_sequence'],
                'advanced_metrics': advanced_metrics,  # Use consistent key name
                'additional_metrics': advanced_metrics  # Keep both for backward compatibility
            }
            
            # Include configuration info if available
            if 'config_info' in analysis_result:
                frame_data['config_info'] = analysis_result['config_info']
                
            session.frame_data = frame_data
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