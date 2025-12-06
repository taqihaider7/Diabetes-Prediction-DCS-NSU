"""
Metrics Collection Module
Tracks ML model inference metrics, latency, throughput, and distributions
"""

import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import os
from dataclasses import dataclass, asdict
import numpy as np
from .shared_metrics import get_shared_store


@dataclass
class PredictionMetric:
    """Single prediction metric data"""
    timestamp: str
    prediction_id: str
    prediction_value: int
    probability: float
    confidence: float
    latency_ms: float
    success: bool
    error_type: Optional[str] = None
    user_session: Optional[str] = None


class MetricsCollector:
    """
    Collects and aggregates ML model metrics
    Thread-safe metrics collection for concurrent predictions
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector
        
        Args:
            retention_hours: How long to retain metrics in memory
        """
        self.retention_hours = retention_hours
        self.lock = threading.Lock()
        
        # Metrics storage
        self.predictions: deque = deque(maxlen=10000)
        self.latencies: deque = deque(maxlen=1000)
        self.errors: List[Dict[str, Any]] = []
        
        # Counters
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        
        # Distribution tracking
        self.prediction_distribution = defaultdict(int)
        self.hourly_counts = defaultdict(int)
        
        # Latency tracking
        self.latency_sum = 0.0
        self.latency_count = 0
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
    def record_prediction(
        self,
        prediction_value: int,
        probability: float,
        confidence: float,
        latency_ms: float,
        prediction_id: Optional[str] = None,
        user_session: Optional[str] = None,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """
        Record a prediction metric
        
        Args:
            prediction_value: Predicted class (0 or 1)
            probability: Probability of positive class
            confidence: Confidence score
            latency_ms: Inference latency in milliseconds
            prediction_id: Unique prediction identifier
            user_session: User session identifier
            success: Whether prediction was successful
            error_type: Type of error if failed
        """
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            metric = PredictionMetric(
                timestamp=timestamp,
                prediction_id=prediction_id or self._generate_id(),
                prediction_value=prediction_value if success else -1,
                probability=probability if success else 0.0,
                confidence=confidence if success else 0.0,
                latency_ms=latency_ms,
                success=success,
                error_type=error_type,
                user_session=user_session
            )
            
            # Store metric
            self.predictions.append(metric)
            
            # Also store in shared storage for cross-process access
            shared_store = get_shared_store()
            shared_store.add_prediction(asdict(metric))
            
            # Update counters
            self.total_predictions += 1
            if success:
                self.successful_predictions += 1
                self.prediction_distribution[prediction_value] += 1
            else:
                self.failed_predictions += 1
                error_data = {
                    'timestamp': timestamp,
                    'error_type': error_type,
                    'prediction_id': metric.prediction_id
                }
                self.errors.append(error_data)
                shared_store.add_error(error_data)
            
            # Track latency
            self.latencies.append(latency_ms)
            self.latency_sum += latency_ms
            self.latency_count += 1
            
            # Track hourly
            hour_key = datetime.now().strftime('%Y-%m-%d %H:00')
            self.hourly_counts[hour_key] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary
        
        Returns:
            Dictionary with all metrics
        """
        with self.lock:
            # Calculate success rate
            success_rate = (
                (self.successful_predictions / self.total_predictions * 100)
                if self.total_predictions > 0 else 0.0
            )
            
            # Calculate latency statistics
            latency_stats = self._calculate_latency_stats()
            
            # Calculate throughput (predictions per minute)
            throughput = self._calculate_throughput()
            
            # Get prediction distribution
            distribution = dict(self.prediction_distribution)
            
            return {
                'total_predictions': self.total_predictions,
                'successful_predictions': self.successful_predictions,
                'failed_predictions': self.failed_predictions,
                'success_rate_percent': round(success_rate, 2),
                'error_rate_percent': round(100 - success_rate, 2),
                'latency': latency_stats,
                'throughput_per_minute': round(throughput, 2),
                'prediction_distribution': distribution,
                'recent_errors': self.errors[-10:] if self.errors else [],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_inference_metrics(self) -> Dict[str, Any]:
        """
        Get detailed inference metrics for monitoring
        
        Returns:
            Inference-specific metrics
        """
        with self.lock:
            latency_stats = self._calculate_latency_stats()
            
            # Get recent predictions (last 100)
            recent_predictions = list(self.predictions)[-100:]
            
            # Calculate distribution percentages
            total = sum(self.prediction_distribution.values())
            distribution_percent = {
                k: round(v / total * 100, 2) if total > 0 else 0
                for k, v in self.prediction_distribution.items()
            }
            
            return {
                'inference_latency': latency_stats,
                'prediction_distribution': dict(self.prediction_distribution),
                'distribution_percentage': distribution_percent,
                'recent_predictions_count': len(recent_predictions),
                'total_inferences': self.total_predictions,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """
        Get error-specific metrics
        
        Returns:
            Error metrics and breakdown
        """
        with self.lock:
            # Count errors by type
            error_breakdown = defaultdict(int)
            for error in self.errors:
                error_type = error.get('error_type', 'unknown')
                error_breakdown[error_type] += 1
            
            # Calculate error rate over time
            recent_errors = [e for e in self.errors 
                           if self._is_recent(e['timestamp'], hours=1)]
            
            return {
                'total_errors': self.failed_predictions,
                'error_rate_percent': round(
                    (self.failed_predictions / self.total_predictions * 100)
                    if self.total_predictions > 0 else 0,
                    2
                ),
                'error_breakdown': dict(error_breakdown),
                'recent_errors_1h': len(recent_errors),
                'latest_errors': self.errors[-5:] if self.errors else [],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_throughput_metrics(self) -> Dict[str, Any]:
        """
        Get throughput and load metrics
        
        Returns:
            Throughput metrics
        """
        with self.lock:
            # Calculate throughput for different time windows
            now = datetime.now()
            
            predictions_1m = self._count_predictions_since(now - timedelta(minutes=1))
            predictions_5m = self._count_predictions_since(now - timedelta(minutes=5))
            predictions_1h = self._count_predictions_since(now - timedelta(hours=1))
            
            return {
                'throughput_1min': predictions_1m,
                'throughput_5min': round(predictions_5m / 5, 2),
                'throughput_1hour': round(predictions_1h / 60, 2),
                'total_requests': self.total_predictions,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent prediction records
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of prediction records
        """
        with self.lock:
            recent = list(self.predictions)[-limit:]
            return [asdict(p) for p in recent]
    
    def export_metrics(self, filepath: Optional[str] = None) -> str:
        """
        Export metrics to JSON file
        
        Args:
            filepath: Path to export file
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            metrics_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(metrics_dir, f'metrics_{timestamp}.json')
        
        with self.lock:
            metrics = {
                'summary': self.get_metrics_summary(),
                'inference': self.get_inference_metrics(),
                'errors': self.get_error_metrics(),
                'throughput': self.get_throughput_metrics(),
                'export_timestamp': datetime.now().isoformat()
            }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return filepath
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.predictions.clear()
            self.latencies.clear()
            self.errors.clear()
            self.total_predictions = 0
            self.successful_predictions = 0
            self.failed_predictions = 0
            self.prediction_distribution.clear()
            self.hourly_counts.clear()
            self.latency_sum = 0.0
            self.latency_count = 0
    
    # Private helper methods
    
    def _calculate_latency_stats(self) -> Dict[str, float]:
        """Calculate latency statistics"""
        if not self.latencies:
            return {
                'mean_ms': 0.0,
                'median_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0
            }
        
        latencies_array = np.array(list(self.latencies))
        return {
            'mean_ms': round(float(np.mean(latencies_array)), 2),
            'median_ms': round(float(np.median(latencies_array)), 2),
            'p95_ms': round(float(np.percentile(latencies_array, 95)), 2),
            'p99_ms': round(float(np.percentile(latencies_array, 99)), 2),
            'min_ms': round(float(np.min(latencies_array)), 2),
            'max_ms': round(float(np.max(latencies_array)), 2)
        }
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput (predictions per minute)"""
        if not self.predictions:
            return 0.0
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        recent_count = sum(
            1 for p in self.predictions
            if self._is_recent(p.timestamp, minutes=1)
        )
        
        return recent_count
    
    def _count_predictions_since(self, since: datetime) -> int:
        """Count predictions since a specific time"""
        count = 0
        for pred in self.predictions:
            pred_time = datetime.fromisoformat(pred.timestamp)
            if pred_time >= since:
                count += 1
        return count
    
    def _is_recent(self, timestamp_str: str, hours: int = 0, minutes: int = 0) -> bool:
        """Check if timestamp is recent"""
        timestamp = datetime.fromisoformat(timestamp_str)
        cutoff = datetime.now() - timedelta(hours=hours, minutes=minutes)
        return timestamp >= cutoff
    
    def _generate_id(self) -> str:
        """Generate unique prediction ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _start_cleanup_thread(self):
        """Start background thread to cleanup old metrics"""
        def cleanup():
            while True:
                time.sleep(3600)  # Run every hour
                self._cleanup_old_metrics()
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=self.retention_hours)
            
            # Clean up errors
            self.errors = [
                e for e in self.errors
                if datetime.fromisoformat(e['timestamp']) >= cutoff
            ]
