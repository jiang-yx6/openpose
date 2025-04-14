import logging
from django.http import JsonResponse

logger = logging.getLogger(__name__)

# Define UI error codes as constants
class ErrorCodes:
    """UI-specific error codes for frontend handling"""
    UNKNOWN_ERROR = 10000
    PROCESSING_ERROR = 10001
    VALIDATION_ERROR = 10002
    
    # Video processing specific errors
    VIDEO_PROCESSING_ERROR = 20000
    FULL_BODY_NOT_VISIBLE = 20001
    VIDEO_TOO_SHORT = 20002
    VIDEO_FORMAT_ERROR = 20003
    POSE_DETECTION_FAILED = 20004
    VIDEO_LENGTH_MISMATCH = 20005
    
    # System errors
    SYSTEM_ERROR = 30000
    DATABASE_ERROR = 30001
    FILE_SYSTEM_ERROR = 30002

class PoseAnalysisError(Exception):
    """Base class for all pose analysis errors"""
    def __init__(self, message, code=500, ui_error_code=ErrorCodes.UNKNOWN_ERROR):
        self.message = message
        self.code = code  # HTTP status code
        self.ui_error_code = ui_error_code  # UI-specific error code
        super().__init__(self.message)

class FullBodyNotVisibleError(PoseAnalysisError):
    """Exception raised when full body is not visible in video"""
    def __init__(self):
        message = "用户上传视频非全身，请录制全身视频"
        super().__init__(message, code=400, ui_error_code=ErrorCodes.FULL_BODY_NOT_VISIBLE)

class VideoLengthMismatchError(PoseAnalysisError):
    """Exception raised when there's an issue with video length or alignment"""
    def __init__(self):
        message = "用户上传视频非全身，请录制全身视频"  # Same message as FullBodyNotVisible for user simplicity
        super().__init__(message, code=400, ui_error_code=ErrorCodes.VIDEO_LENGTH_MISMATCH)

class ApiErrorHandler:
    """Centralized error handling for API endpoints"""
    
    @staticmethod
    def is_inhomogeneous_shape_error(error):
        """Check if a ValueError is related to inhomogeneous shapes in numpy array"""
        error_str = str(error)
        return ("inhomogeneous shape" in error_str or 
                "setting an array element with a sequence" in error_str)
    
    @staticmethod
    def handle_exception(exception, session=None):
        """Handle exceptions and return appropriate response"""
        # Convert specific errors to our custom exceptions
        if isinstance(exception, ValueError) and ApiErrorHandler.is_inhomogeneous_shape_error(exception):
            exception = FullBodyNotVisibleError()
        elif isinstance(exception, IndexError) and "list index out of range" in str(exception):
            exception = VideoLengthMismatchError()
        
        # Update session if provided
        if session:
            session.status = 'failed'
            session.error_message = str(exception)
            session.save()
        
        # Log the error
        logger.warning(f"视频处理失败: {str(exception)}")
        
        # Create response based on exception type
        if isinstance(exception, PoseAnalysisError):
            return JsonResponse({
                'error': exception.message,
                'status': 'failed',
                'error_code': exception.code,  # HTTP status code
                'ui_error_code': exception.ui_error_code  # UI-specific error code
            }, status=exception.code)
        else:
            # Default error handling for unknown exceptions
            return JsonResponse({
                'error': str(exception),
                'status': 'failed',
                'error_code': 500,
                'ui_error_code': ErrorCodes.UNKNOWN_ERROR
            }, status=500)