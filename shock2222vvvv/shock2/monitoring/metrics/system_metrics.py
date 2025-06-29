"""
Shock2 System Metrics
Comprehensive system monitoring and metrics collection
"""

import psutil
import time
import logging
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """Complete system snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    load_average: List[float]
    uptime_seconds: float


class SystemMetrics:
    """System metrics collector and monitor"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.collection_interval = config.get('collection_interval', 5.0)
        self.retention_hours = config.get('retention_hours', 24)
        self.alert_thresholds = config.get('alert_thresholds', {})

        # Metrics storage
        self.metrics_history: Dict[str, deque] = {
            'cpu_percent': deque(maxlen=720),  # 1 hour at 5s intervals
            'memory_percent': deque(maxlen=720),
            'disk_percent': deque(maxlen=720),
            'network_io': deque(maxlen=720),
            'process_count': deque(maxlen=720)
        }

        # System snapshots
        self.snapshots = deque(maxlen=720)

        # Alert state
        self.alert_states = {}
        self.last_alerts = {}

        # Collection state
        self.is_collecting = False
        self.collection_task = None

        # Performance tracking
        self.collection_stats = {
            'total_collections': 0,
            'failed_collections': 0,
            'avg_collection_time': 0.0,
            'last_collection': None
        }

        # Network baseline
        self.network_baseline = None
        self.last_network_stats = None

    async def start_collection(self):
        """Start metrics collection"""
        if self.is_collecting or not self.enabled:
            return

        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())

        # Initialize network baseline
        self._initialize_network_baseline()

        logger.info("? System metrics collection started")

    async def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False

        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        logger.info("? System metrics collection stopped")

    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.is_collecting:
            try:
                start_time = time.time()

                # Collect system snapshot
                snapshot = await self._collect_system_snapshot()

                # Store snapshot
                self.snapshots.append(snapshot)

                # Store individual metrics
                self._store_metrics(snapshot)

                # Check alerts
                await self._check_alerts(snapshot)

                # Update collection stats
                collection_time = time.time() - start_time
                self._update_collection_stats(True, collection_time)

                # Wait for next collection
                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                self._update_collection_stats(False, 0)
                await asyncio.sleep(self.collection_interval)

    async def _collect_system_snapshot(self) -> SystemSnapshot:
        """Collect complete system snapshot"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024 ** 3)
        memory_total_gb = memory.total / (1024 ** 3)

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_used_gb = disk.used / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)

        # Network metrics
        network_sent_mb, network_recv_mb = self._get_network_rates()

        # Process metrics
        process_count = len(psutil.pids())

        # Load average (Unix-like systems)
        try:
            load_average = list(os.getloadavg())
        except (AttributeError, OSError):
            load_average = [0.0, 0.0, 0.0]

        # System uptime
        uptime_seconds = time.time() - psutil.boot_time()

        return SystemSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_count=process_count,
            load_average=load_average,
            uptime_seconds=uptime_seconds
        )

    def _initialize_network_baseline(self):
        """Initialize network statistics baseline"""
        try:
            self.network_baseline = psutil.net_io_counters()
            self.last_network_stats = self.network_baseline
        except Exception as e:
            logger.warning(f"Failed to initialize network baseline: {e}")
            self.network_baseline = None

    def _get_network_rates(self) -> tuple[float, float]:
        """Get network send/receive rates in MB/s"""
        if not self.network_baseline or not self.last_network_stats:
            return 0.0, 0.0

        try:
            current_stats = psutil.net_io_counters()

            # Calculate rates
            time_delta = self.collection_interval
            sent_delta = current_stats.bytes_sent - self.last_network_stats.bytes_sent
            recv_delta = current_stats.bytes_recv - self.last_network_stats.bytes_recv

            sent_rate_mb = (sent_delta / time_delta) / (1024 ** 2)
            recv_rate_mb = (recv_delta / time_delta) / (1024 ** 2)

            # Update last stats
            self.last_network_stats = current_stats

            return sent_rate_mb, recv_rate_mb

        except Exception as e:
            logger.warning(f"Failed to calculate network rates: {e}")
            return 0.0, 0.0

    def _store_metrics(self, snapshot: SystemSnapshot):
        """Store metrics in history"""
        timestamp = snapshot.timestamp

        # Store individual metrics
        self.metrics_history['cpu_percent'].append(
            MetricPoint(timestamp, snapshot.cpu_percent)
        )

        self.metrics_history['memory_percent'].append(
            MetricPoint(timestamp, snapshot.memory_percent)
        )

        self.metrics_history['disk_percent'].append(
            MetricPoint(timestamp, snapshot.disk_percent)
        )

        self.metrics_history['network_io'].append(
            MetricPoint(timestamp, snapshot.network_sent_mb + snapshot.network_recv_mb)
        )

        self.metrics_history['process_count'].append(
            MetricPoint(timestamp, snapshot.process_count)
        )

    async def _check_alerts(self, snapshot: SystemSnapshot):
        """Check for alert conditions"""
        alerts_triggered = []

        # CPU alert
        cpu_threshold = self.alert_thresholds.get('cpu_usage', 80)
        if snapshot.cpu_percent > cpu_threshold:
            alert = await self._trigger_alert('high_cpu_usage', {
                'current': snapshot.cpu_percent,
                'threshold': cpu_threshold,
                'severity': 'warning' if snapshot.cpu_percent < 90 else 'critical'
            })
            if alert:
                alerts_triggered.append(alert)

        # Memory alert
        memory_threshold = self.alert_thresholds.get('memory_usage', 85)
        if snapshot.memory_percent > memory_threshold:
            alert = await self._trigger_alert('high_memory_usage', {
                'current': snapshot.memory_percent,
                'threshold': memory_threshold,
                'severity': 'warning' if snapshot.memory_percent < 95 else 'critical'
            })
            if alert:
                alerts_triggered.append(alert)

        # Disk alert
        disk_threshold = self.alert_thresholds.get('disk_usage', 90)
        if snapshot.disk_percent > disk_threshold:
            alert = await self._trigger_alert('high_disk_usage', {
                'current': snapshot.disk_percent,
                'threshold': disk_threshold,
                'severity': 'warning' if snapshot.disk_percent < 95 else 'critical'
            })
            if alert:
                alerts_triggered.append(alert)

        # Load average alert (Unix-like systems)
        if snapshot.load_average[0] > 0:
            cpu_count = psutil.cpu_count()
            load_threshold = cpu_count * 0.8
            if snapshot.load_average[0] > load_threshold:
                alert = await self._trigger_alert('high_load_average', {
                    'current': snapshot.load_average[0],
                    'threshold': load_threshold,
                    'cpu_count': cpu_count,
                    'severity': 'warning'
                })
                if alert:
                    alerts_triggered.append(alert)

        return alerts_triggered

    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Trigger an alert with rate limiting"""
        current_time = datetime.now()

        # Rate limiting - don't send same alert more than once per 5 minutes
        last_alert_time = self.last_alerts.get(alert_type)
        if last_alert_time and (current_time - last_alert_time).total_seconds() < 300:
            return None

        alert = {
            'type': alert_type,
            'timestamp': current_time.isoformat(),
            'data': data,
            'severity': data.get('severity', 'warning')
        }

        # Log alert
        severity = data.get('severity', 'warning')
        if severity == 'critical':
            logger.critical(f"? CRITICAL ALERT: {alert_type} - {data}")
        else:
            logger.warning(f"?? WARNING: {alert_type} - {data}")

        # Update alert state
        self.alert_states[alert_type] = alert
        self.last_alerts[alert_type] = current_time

        return alert

    def _update_collection_stats(self, success: bool, collection_time: float):
        """Update collection statistics"""
        self.collection_stats['total_collections'] += 1
        self.collection_stats['last_collection'] = datetime.now().isoformat()

        if success:
            # Update average collection time
            total = self.collection_stats['total_collections']
            current_avg = self.collection_stats['avg_collection_time']
            self.collection_stats['avg_collection_time'] = (
                    (current_avg * (total - 1) + collection_time) / total
            )
        else:
            self.collection_stats['failed_collections'] += 1

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if not self.snapshots:
            return {}

        latest = self.snapshots[-1]

        return {
            'timestamp': latest.timestamp.isoformat(),
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'memory_used_gb': round(latest.memory_used_gb, 2),
            'memory_total_gb': round(latest.memory_total_gb, 2),
            'disk_percent': latest.disk_percent,
            'disk_used_gb': round(latest.disk_used_gb, 2),
            'disk_total_gb': round(latest.disk_total_gb, 2),
            'network_sent_mb': round(latest.network_sent_mb, 2),
            'network_recv_mb': round(latest.network_recv_mb, 2),
            'process_count': latest.process_count,
            'load_average': latest.load_average,
            'uptime_hours': round(latest.uptime_seconds / 3600, 2)
        }

    def get_metrics_history(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics for specified time period"""
        if metric_name not in self.metrics_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = self.metrics_history[metric_name]

        return [
            {
                'timestamp': point.timestamp.isoformat(),
                'value': point.value,
                'metadata': point.metadata
            }
            for point in history
            if point.timestamp >= cutoff_time
        ]

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        current = self.get_current_metrics()

        # Calculate averages over last hour
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= hour_ago]

        if recent_snapshots:
            avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
            avg_memory = sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots)
            avg_disk = sum(s.disk_percent for s in recent_snapshots) / len(recent_snapshots)
        else:
            avg_cpu = avg_memory = avg_disk = 0

        return {
            'current': current,
            'averages_1h': {
                'cpu_percent': round(avg_cpu, 2),
                'memory_percent': round(avg_memory, 2),
                'disk_percent': round(avg_disk, 2)
            },
            'collection_stats': self.collection_stats,
            'active_alerts': list(self.alert_states.keys()),
            'snapshots_count': len(self.snapshots),
            'is_collecting': self.is_collecting
        }

    async def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics to JSON file"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'export_period_hours': hours,
            'snapshots': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'cpu_percent': s.cpu_percent,
                    'memory_percent': s.memory_percent,
                    'memory_used_gb': s.memory_used_gb,
                    'memory_total_gb': s.memory_total_gb,
                    'disk_percent': s.disk_percent,
                    'disk_used_gb': s.disk_used_gb,
                    'disk_total_gb': s.disk_total_gb,
                    'network_sent_mb': s.network_sent_mb,
                    'network_recv_mb': s.network_recv_mb,
                    'process_count': s.process_count,
                    'load_average': s.load_average,
                    'uptime_seconds': s.uptime_seconds
                }
                for s in self.snapshots
                if s.timestamp >= cutoff_time
            ],
            'summary': self.get_system_summary()
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"? Exported {len(export_data['snapshots'])} metrics snapshots to {filepath}")

        return len(export_data['snapshots'])
