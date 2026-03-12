"""
Monitoring and metrics collection module
"""

from datetime import datetime
from typing import Dict
import time

class MetricsCollector:
    """Collect and track API metrics"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.predictions_by_emotion = {}
    
    def record_request(self):
        """Record incoming request"""
        self.request_count += 1
    
    def record_error(self):
        """Record error"""
        self.error_count += 1
    
    def record_processing_time(self, duration: float):
        """Record processing time"""
        self.total_processing_time += duration
    
    def record_emotion_prediction(self, emotion: str):
        """Record emotion prediction"""
        self.predictions_by_emotion[emotion] = \
            self.predictions_by_emotion.get(emotion, 0) + 1
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        uptime = datetime.now() - self.start_time
        avg_processing_time = (self.total_processing_time / self.request_count 
                              if self.request_count > 0 else 0)
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": (self.error_count / self.request_count 
                          if self.request_count > 0 else 0),
            "avg_processing_time": avg_processing_time,
            "predictions_by_emotion": self.predictions_by_emotion
        }
    
    def reset(self):
        """Reset metrics"""
        self.__init__()

# Global metrics instance
metrics = MetricsCollector()
