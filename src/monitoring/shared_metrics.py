"""
Shared Metrics Storage
Allows multiple Streamlit apps to share metrics data through file-based storage
"""

import json
import threading
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import time


class SharedMetricsStore:
    """
    File-based metrics storage that can be shared between multiple processes
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize shared metrics store
        
        Args:
            storage_dir: Directory to store metrics file
        """
        if storage_dir is None:
            # Use absolute path to ensure both apps use the same location
            storage_dir = Path(__file__).parent.parent.parent / 'metrics'
        else:
            storage_dir = Path(storage_dir)
        
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = storage_dir / 'shared_metrics.json'
        self.lock_file = storage_dir / 'shared_metrics.lock'
        self._lock = threading.Lock()
        
        # Initialize file if it doesn't exist
        if not self.metrics_file.exists():
            self._write_metrics({
                'predictions': [],
                'errors': [],
                'last_updated': datetime.now().isoformat()
            })
    
    def _read_metrics(self) -> Dict[str, Any]:
        """Read metrics from file"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        # Return default structure
        return {
            'predictions': [],
            'errors': [],
            'last_updated': datetime.now().isoformat()
        }
    
    def _write_metrics(self, data: Dict[str, Any]):
        """Write metrics to file"""
        with self._lock:
            data['last_updated'] = datetime.now().isoformat()
            try:
                with open(self.metrics_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Error writing metrics: {e}")
    
    def add_prediction(self, prediction_data: Dict[str, Any]):
        """Add a prediction record"""
        data = self._read_metrics()
        data['predictions'].append(prediction_data)
        
        # Keep only recent predictions (last 24 hours by default)
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        data['predictions'] = [
            p for p in data['predictions']
            if p.get('timestamp', '') > cutoff
        ]
        
        self._write_metrics(data)
    
    def add_error(self, error_data: Dict[str, Any]):
        """Add an error record"""
        data = self._read_metrics()
        data['errors'].append(error_data)
        
        # Keep only recent errors (last 24 hours)
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        data['errors'] = [
            e for e in data['errors']
            if e.get('timestamp', '') > cutoff
        ]
        
        self._write_metrics(data)
    
    def get_all_predictions(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get all predictions"""
        data = self._read_metrics()
        predictions = sorted(
            data.get('predictions', []),
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        if limit:
            return predictions[:limit]
        return predictions
    
    def get_all_errors(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get all errors"""
        data = self._read_metrics()
        errors = sorted(
            data.get('errors', []),
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        if limit:
            return errors[:limit]
        return errors
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        data = self._read_metrics()
        predictions = data.get('predictions', [])
        errors = data.get('errors', [])
        
        successful = [p for p in predictions if p.get('success', False)]
        failed = [p for p in predictions if not p.get('success', False)]
        
        # Calculate latency stats
        latencies = [p['latency_ms'] for p in successful if 'latency_ms' in p]
        
        if latencies:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            latency_stats = {
                'mean_ms': sum(latencies) / n,
                'median_ms': latencies_sorted[n // 2],
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'p95_ms': latencies_sorted[int(n * 0.95)] if n > 0 else 0,
                'p99_ms': latencies_sorted[int(n * 0.99)] if n > 0 else 0,
            }
        else:
            latency_stats = {
                'mean_ms': 0, 'median_ms': 0, 'min_ms': 0,
                'max_ms': 0, 'p95_ms': 0, 'p99_ms': 0
            }
        
        # Prediction distribution
        distribution = {}
        for p in successful:
            val = p.get('prediction_value', -1)
            distribution[val] = distribution.get(val, 0) + 1
        
        # Throughput
        now = datetime.now()
        recent_1min = [p for p in predictions if (now - datetime.fromisoformat(p['timestamp'])).total_seconds() <= 60]
        recent_5min = [p for p in predictions if (now - datetime.fromisoformat(p['timestamp'])).total_seconds() <= 300]
        recent_1hour = [p for p in predictions if (now - datetime.fromisoformat(p['timestamp'])).total_seconds() <= 3600]
        
        return {
            'total_predictions': len(predictions),
            'successful_predictions': len(successful),
            'failed_predictions': len(failed),
            'success_rate_percent': (len(successful) / len(predictions) * 100) if predictions else 0,
            'total_errors': len(errors),
            'latency': latency_stats,
            'prediction_distribution': distribution,
            'throughput': {
                'throughput_1min': len(recent_1min),
                'throughput_5min': len(recent_5min) / 5,
                'throughput_1hour': len(recent_1hour) / 60
            }
        }
    
    def clear_all(self):
        """Clear all metrics"""
        self._write_metrics({
            'predictions': [],
            'errors': [],
            'last_updated': datetime.now().isoformat()
        })


# Global instance
_shared_store = None

def get_shared_store() -> SharedMetricsStore:
    """Get or create global shared metrics store"""
    global _shared_store
    if _shared_store is None:
        _shared_store = SharedMetricsStore()
    return _shared_store
