import logging
from .services import VideoProcessingService
from .modern_pose_analyzer import ModernPoseAnalyzer

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
    Wrapper around the existing VideoProcessingService that uses the modern pose analyzer.
    """
    
    def __init__(self):
        super().__init__()
        self.modern_analyzer = ModernPoseAnalyzer()
        
    def process_videos(self, session_id, standard_video_path, exercise_video_path):
        """
        Process videos using the modern implementation while maintaining the
        same interface as the original VideoProcessingService.
        """
        try:
            # Use modern analyzer to process videos
            modern_result = self.modern_analyzer.process_videos(
                session_id, 
                standard_video_path, 
                exercise_video_path
            )
            
            if not modern_result['dtw_success']:
                raise Exception(f"Modern pose analysis failed: {modern_result.get('error', 'Unknown error')}")
            
            # Get sequences and DTW result
            std_sequence = modern_result['std_sequence']
            exe_sequence = modern_result['exe_sequence']
            dtw_result = modern_result['dtw_result']
            
            # Use the rest of the VideoProcessingService functionality
            # by setting the results on the session and generating videos
            from .models import EvalSession
            session = EvalSession.objects.get(pk=session_id)
            session.dtw_distance = modern_result['dtw_distance']
            session.similarity_score = modern_result['similarity_score']
            session.frame_scores = modern_result['frame_scores']
            session.frame_data = {'std_frame_data': std_sequence, 'exercise_frame_data': exe_sequence}
            session.status = 'completed'
            session.save()
            
            # Now continue with the video processing using the parent class implementation
            # but use our modern overlap video processor
            result = super().process_videos(session_id, standard_video_path, exercise_video_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Modern video processing failed: {str(e)}", exc_info=True)
            raise