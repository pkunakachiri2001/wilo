"""
Event Manager Module
Handles logging of failure events with multi-sensor trend extraction.
Extracts trends based on aggregated features (mean, max, std_dev, kurtosis) across all sensors.
Tracks slope changes BACKWARDS from failure point for acceleration, current, and audio.
Queries aggregated data from database tables (acceleration, current, audio).
"""

import os
import csv
import json
import datetime
import numpy as np
from scipy import stats as sp_stats
from typing import Dict, List, Tuple, Optional


def _get_expected_frequencies(fault_name: str) -> List[float]:
    name = (fault_name or '').lower()
    f_shaft = 730.0 / 60.0 # 12.17
    line = 50.0
    if 'bearing' in name:
        return [29.4, 43.6, 11.8, f_shaft]
    elif 'misalignment' in name:
        return [f_shaft, 2 * f_shaft]
    elif 'overheating' in name:
        return [2 * line]
    elif 'stall' in name:
        return [2 * line]
    elif 'winding' in name:
        return [3 * line, 5 * line, line]
    elif 'electrical' in name:
        return [2 * line]
    elif 'vibration' in name:
        return [0.5 * f_shaft, f_shaft, 1.5 * f_shaft, 2 * f_shaft]
    elif 'cavitation' in name:
        return [5 * f_shaft]
    elif 'impeller' in name:
        return [5 * f_shaft, 4 * f_shaft, 6 * f_shaft]
    elif 'seal' in name:
        return [5 * f_shaft]
    else:
        return []


def _calculate_fault_frequency_match(freqs: List[float], expected: List[float], tolerance: float = 0.10) -> float:
    if not expected or not freqs:
        return 0.0
    matched = 0
    for f in freqs:
        if f == 0:
            continue
        for exp_f in expected:
            if abs(f - exp_f) / exp_f <= tolerance:
                matched += 1
                break
    return float(matched) / len(freqs)


class EventManager:
    def __init__(self, events_dir: str, data_dir: str):
        """
        Initialize the Event Manager.
        
        Args:
            events_dir: Directory to store event CSVs (only on local, not on Render)
            data_dir: Directory containing max_reading CSV files
        """
        self.events_dir = events_dir
        self.data_dir = data_dir

    def get_next_fault_id_local(self, event_name: str) -> int:
        """
        Scan existing metadata files inside Events/<event_name>/ and also matching files in Events/
        to find the max fault_id / fault_id_in_database, and return the next ID (starting at 1 if none exist).
        """
        import glob
        import json
        
        max_id = 0
        
        # 1. Scan subdirectory Events/<event_name>/
        target_dir = os.path.join(self.events_dir, event_name)
        if os.path.exists(target_dir):
            json_files = glob.glob(os.path.join(target_dir, '*.json'))
            for fpath in json_files:
                basename = os.path.basename(fpath)
                if basename in ('stats.json', 'metadata.json'):
                    continue
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        fid = data.get('fault_id') or data.get('fault_id_in_database')
                        if fid is not None:
                            max_id = max(max_id, int(fid))
                except Exception:
                    continue
                    
        # 2. Scan parent directory Events/ for matching filenames (e.g. Motor_Stall_*.json)
        event_name_safe = event_name.replace(' ', '_').replace('/', '-')
        parent_json_files = glob.glob(os.path.join(self.events_dir, f"{event_name_safe}_*.json"))
        for fpath in parent_json_files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    fid = data.get('fault_id') or data.get('fault_id_in_database')
                    if fid is not None:
                        max_id = max(max_id, int(fid))
            except Exception:
                continue
                
        return max_id + 1
    
    def _load_all_data_points(self) -> List[Tuple[float, float]]:
        """
        Load all data points from max_reading CSV files.
        
        Returns:
            List of (timestamp_ms, z_value, filename) tuples sorted by timestamp
        """
        import glob
        
        all_points = []
        max_reading_files = glob.glob(os.path.join(self.data_dir, 'max_reading*.csv'))
        
        for file_path in max_reading_files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            # Handle different timestamp formats
                            timestamp_str = row['timestamp']
                            if 'T' in timestamp_str:  # ISO format
                                dt_obj = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                timestamp_ms = dt_obj.timestamp() * 1000
                            else:
                                timestamp_ms = float(timestamp_str)
                            
                            # Handle different value column names
                            if 'z' in row:
                                z_value = float(row['z'])
                            elif 'value' in row:
                                z_value = float(row['value'])
                            else:
                                continue
                            
                            all_points.append((timestamp_ms, z_value, filename))
                        except (ValueError, KeyError):
                            continue
            except Exception as e:
                print(f"Error loading data from {file_path}: {e}")
                continue
        
        # Sort by timestamp
        all_points.sort(key=lambda x: x[0])
        return all_points
    
    # -- Fault classification for intelligent deviation detection ----------------
    # SUDDEN: flat baseline until one catastrophic interval; deviation IS the failure.
    # GRADUAL: progressive degradation; 3-sigma scan finds the onset reliably.
    FAULT_TYPES = {
        'Motor Stall':            'SUDDEN',
        'Pump Cavitation':        'SUDDEN',
        'Pump Impeller Damage':   'SUDDEN',
        'Pump Seal Leakage':      'SUDDEN',
        'Motor Bearing Failure':  'GRADUAL',
        'Motor Shaft Misalignment': 'GRADUAL',
        'Motor Overheating':      'GRADUAL',
        'Motor Winding Failure':  'GRADUAL',
        'Motor Vibration Anomaly': 'GRADUAL',
        'Motor Electrical Fault': 'GRADUAL',
        'Custom Event':           'GRADUAL',
    }

    def _load_all_sensor_data(self, start_time_iso: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Load aggregated feature data for ALL sensors from local JSONL files in self.data_dir.
        If start_time_iso is provided, fetches only records created since then (representing the current generator run).
        Otherwise, defaults to fetching the last 30 MAX-mode rows per sensor.

        Returns:
            Dict mapping sensor names to lists of feature data:
            {
                'acceleration': [...],
                'current': [...],
                'audio': [...]
            }
        """
        import logging
        import datetime
        logger = logging.getLogger(__name__)

        sensor_data = {'acceleration': [], 'current': [], 'audio': []}
        
        # Calculate safety buffered start time for clock drift / desync
        query_start_time = None
        if start_time_iso:
            try:
                dt_start = datetime.datetime.fromisoformat(start_time_iso.replace('Z', '+00:00'))
                query_start_time = dt_start - datetime.timedelta(seconds=5)
                logger.info(f"🔍 Buffered start_time={query_start_time} (original={start_time_iso})")
            except Exception as e:
                logger.warning(f"Error parsing start_time_iso: {e}. Using raw start_time_iso.")
                query_start_time = start_time_iso
        else:
            LIMIT = 30
            logger.info(f"🔍 Loading last {LIMIT} MAX rows per sensor")

        logger.info(f"💾 [LOCAL] Loading sensor data from local JSONL files in {self.data_dir}")
        
        for sensor_type in ('acceleration', 'current', 'audio'):
            file_path = os.path.join(self.data_dir, f"{sensor_type}.jsonl")
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ Local file not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                rows = []
                for line in lines:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get('file_type') != 'max':
                        continue
                        
                    # Parse time
                    time_str = row.get('created_at') or row.get('timestamp')
                    if not time_str:
                        continue
                    try:
                        dt = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    except Exception as te:
                        logger.error(f"Error parsing timestamp {time_str}: {te}")
                        continue
                        
                    # Filter by start_time if query_start_time is present
                    target_dt = query_start_time
                    if isinstance(target_dt, str):
                        try:
                            target_dt = datetime.datetime.fromisoformat(target_dt.replace('Z', '+00:00'))
                        except:
                            target_dt = None
                    
                    if target_dt:
                        if dt.tzinfo is None and target_dt.tzinfo is not None:
                            dt = dt.replace(tzinfo=datetime.timezone.utc)
                        elif dt.tzinfo is not None and target_dt.tzinfo is None:
                            target_dt = target_dt.replace(tzinfo=datetime.timezone.utc)
                        
                        if dt < target_dt:
                            continue
                    
                    std_dev_val = row.get('standard_deviation') if row.get('standard_deviation') is not None else 0.0
                    feature_data = {
                        'timestamp': dt.timestamp() * 1000,
                        'min': row.get('x_min', 0.0),
                        'max': row.get('x_max', 0.0),
                        'mean': row.get('mean', 0.0),
                        'std_dev': std_dev_val,
                        'range': row.get('range', 0.0),
                        'variance': std_dev_val ** 2,
                        'skewness': row.get('skewness', 0.0),
                        'kurtosis': row.get('kurtosis', 0.0),
                        'rms': row.get('rms', 0.0),
                        'peak': row.get('peak', 0.0),
                        'crest_factor': row.get('crest_factor', 0.0),
                        'load_factor': row.get('load_factor', 1.0),
                        'frequency1': row.get('frequency1', 0.0),
                        'frequency2': row.get('frequency2', 0.0),
                        'frequency3': row.get('frequency3', 0.0),
                        'frequency4': row.get('frequency4', 0.0),
                        'frequency5': row.get('frequency5', 0.0),
                        'amplitude1': row.get('amplitude1', 0.0),
                        'amplitude2': row.get('amplitude2', 0.0),
                        'amplitude3': row.get('amplitude3', 0.0),
                        'amplitude4': row.get('amplitude4', 0.0),
                        'amplitude5': row.get('amplitude5', 0.0),
                    }
                    rows.append(feature_data)
                
                rows.sort(key=lambda x: x['timestamp'])
                
                if not query_start_time:
                    rows = rows[-LIMIT:]
                    
                sensor_data[sensor_type] = rows
                logger.info(f"✓ Loaded {len(sensor_data[sensor_type])} {sensor_type} records from local file")
            except Exception as file_err:
                logger.error(f"Error loading local data for {sensor_type}: {file_err}")
        
        return sensor_data
    
    def _find_nearest_data_point(self, failure_time_ms: float, data_points: List[Tuple[float, float]]) -> Optional[int]:
        """
        Find index of data point at or just before the failure time.
        
        Args:
            failure_time_ms: Failure timestamp in milliseconds
            data_points: List of (timestamp, value) tuples
            
        Returns:
            Index of nearest data point at or before failure, or None if no data available
        """
        if not data_points:
            return None
        
        # Find the last point that is at or before the failure time
        nearest_idx = None
        
        for idx, point in enumerate(data_points):
            timestamp = point[0]
            if timestamp <= failure_time_ms:
                nearest_idx = idx
            else:
                break  # Stop when we find a point after failure
        
        return nearest_idx

    # _calculate_feature_slopes removed - slope computation is done inline
    # inside _extract_multi_sensor_trends with correct forward/backward differences.
    #
    # _find_stable_baseline_idx removed - was a stub returning last index;
    # deviation onset is now detected dynamically in _extract_multi_sensor_trends.
    
    # Minimum number of chronological MAX-mode upload cycles required to attempt
    # any trend extraction.  With fewer points there is no baseline to compare
    # against, deviation detection is impossible, and all slopes are 0.
    MIN_TREND_POINTS = 5

    # Fault-specific parameter importance weights.
    # Derived from physical signatures — the same table as trend_extractor.py.
    # Format: {fault_name_lower: [(sensor, feature, weight, direction), ...]}
    # direction: 'up' (only positive z counts), 'down', 'abs' (magnitude)
    _FAULT_WEIGHTS = {
        'motor bearing failure':   [
            ('acceleration', 'kurtosis',  1.0, 'up'),
            ('acceleration', 'skewness',  0.7, 'abs'),
            ('acceleration', 'std_dev',   0.6, 'up'),
            ('audio',        'kurtosis',  0.7, 'up'),
        ],
        'motor electrical fault':  [
            ('current',      'skewness',  1.0, 'abs'),
            ('current',      'kurtosis',  0.9, 'up'),
            ('current',      'std_dev',   0.7, 'up'),
            ('acceleration', 'std_dev',   0.4, 'up'),
        ],
        'motor overheating':       [
            ('current',      'mean',      1.0, 'up'),
            ('current',      'std_dev',   0.7, 'up'),
            ('audio',        'mean',      0.4, 'up'),
        ],
        'motor shaft misalignment':[
            ('acceleration', 'mean',      1.0, 'up'),
            ('acceleration', 'std_dev',   0.8, 'up'),
            ('current',      'std_dev',   0.5, 'up'),
        ],
        'motor stall':             [
            ('current',      'mean',      1.0, 'up'),
            ('acceleration', 'mean',      1.0, 'down'),
            ('acceleration', 'range',     0.8, 'down'),
        ],
        'motor vibration anomaly': [
            ('acceleration', 'std_dev',   1.0, 'up'),
            ('acceleration', 'kurtosis',  0.9, 'up'),
            ('acceleration', 'mean',      0.7, 'up'),
            ('audio',        'std_dev',   0.5, 'up'),
        ],
        'motor winding failure':   [
            ('current',      'kurtosis',  1.0, 'up'),
            ('current',      'skewness',  0.9, 'abs'),
            ('current',      'std_dev',   0.8, 'up'),
            ('acceleration', 'std_dev',   0.3, 'up'),
        ],
        'pump cavitation':         [
            ('acceleration', 'kurtosis',  1.0, 'up'),
            ('acceleration', 'skewness',  0.8, 'abs'),
            ('acceleration', 'range',     0.7, 'up'),
            ('audio',        'kurtosis',  0.9, 'up'),
            ('audio',        'std_dev',   0.6, 'up'),
        ],
        'pump impeller damage':    [
            ('acceleration', 'std_dev',   1.0, 'up'),
            ('acceleration', 'kurtosis',  0.8, 'up'),
            ('current',      'std_dev',   0.6, 'up'),
        ],
        'pump seal leakage':       [
            ('audio',        'std_dev',   1.0, 'up'),
            ('audio',        'mean',      0.8, 'up'),
            ('current',      'mean',      0.5, 'down'),  # current drops on seal leak
        ],
        'custom event':            [
            ('acceleration', 'kurtosis',  1.0, 'abs'),
            ('acceleration', 'std_dev',   1.0, 'abs'),
            ('current',      'kurtosis',  1.0, 'abs'),
            ('audio',        'kurtosis',  1.0, 'abs'),
        ],
    }
    _DEFAULT_WEIGHTS = [
        ('acceleration', 'kurtosis', 1.0, 'abs'),
        ('acceleration', 'std_dev',  1.0, 'abs'),
        ('current',      'kurtosis', 1.0, 'abs'),
        ('audio',        'kurtosis', 1.0, 'abs'),
    ]

    @staticmethod
    def _weighted_cusum_deviation_score(
        record: Dict,
        baseline_stats: Dict,
        weights: List[tuple],
        min_z: float = 0.8,
    ) -> float:
        """
        Compute a weighted deviation score for a single data record.

        Score = Σ (weight × max(0, |z_score| − min_z)) / Σ weights
                 where z_score = (val − baseline_mean) / max(baseline_std, 1e-9)
                 and direction filter applies ('up'/'down'/'abs')

        min_z=0.8 (not 3.0) means even a 0.8σ deviation contributes once
        weighted — this lets CUSUM accumulate small early-stage changes
        that a single 3σ threshold would miss.
        """
        total_w = sum(w for _, _, w, _ in weights)
        if total_w == 0:
            return 0.0
        score = 0.0
        for sensor, feature, weight, direction in weights:
            val = record.get(sensor, {}).get(feature) if isinstance(record.get(sensor), dict) \
                  else record.get(feature)
            # Flat dict fallback (sensor_data rows are flat dicts)
            if val is None:
                val = record.get(feature)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            bl = baseline_stats.get(sensor, {}).get(feature)
            if bl is None:
                continue
            b_mean, b_std = bl
            b_std = max(b_std, 1e-9)
            z = (val - b_mean) / b_std
            if direction == 'up'   and z < 0: continue
            if direction == 'down' and z > 0: continue
            score += weight * max(0.0, abs(z) - min_z)
        return score / total_w

    def _extract_multi_sensor_trends(
        self,
        sensor_data: Dict,
        failure_time_ms: Optional[float] = None,
        fault_name: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Extract trend data for all sensors with physics-grounded onset detection.

        GRADUAL faults — CUSUM (Cumulative Sum) detector:
            Accumulates weighted deviation scores per interval.
            CUSUM_pos[i] = max(0, CUSUM_pos[i-1] + score[i] − k)
            where k = 0.5 (slack parameter — ignores very small deviations).
            Triggers when CUSUM_pos > h (decision threshold = 3.0).

            Why CUSUM over 3σ:
              • A bearing in Stage 2 shows kurtosis 3→7 — only +1.3σ individually,
                but accumulated over 4 intervals × weight 1.0 it easily exceeds h=3.0.
              • 3σ misses this entirely until Stage 3/4.

        SUDDEN faults — deviation IS the failure interval:
            No accumulation needed. deviation_idx = failure_idx directly.

        Baseline:
            First 25% of intervals (minimum 3, maximum 20) — more robust than
            the old 20%-minimum-5 approach when Stage 2 onset is early.

        Raises ValueError if any sensor has fewer than MIN_TREND_POINTS data points.
        """
        import logging as _logging
        import numpy as np
        _log = _logging.getLogger(__name__)

        ALL_FEATURES = ['mean', 'max', 'min', 'std_dev', 'range', 'variance', 'skewness', 'kurtosis', 'rms', 'peak', 'crest_factor']

        # Classify fault
        fault_category = self.FAULT_TYPES.get(fault_name or '', 'GRADUAL')
        _log.info(f"🎯 Fault '{fault_name}' → {fault_category}")

        # Minimum data guard
        insufficient = {
            s: len(sensor_data.get(s, []))
            for s in ['acceleration', 'current', 'audio']
            if len(sensor_data.get(s, [])) < self.MIN_TREND_POINTS
        }
        if insufficient:
            detail = ", ".join(f"{s}: {n} row(s)" for s, n in insufficient.items())
            raise ValueError(
                f"Insufficient data for trend extraction ({detail}). "
                f"Need at least {self.MIN_TREND_POINTS} chronological MAX-mode upload "
                f"cycles per sensor."
            )

        accel_data = sensor_data.get('acceleration', [])
        if not accel_data:
            raise ValueError("No acceleration data available")

        # Locate failure index
        if failure_time_ms is not None:
            failure_idx = self._find_nearest_data_point(
                failure_time_ms, [(p['timestamp'], 0.0) for p in accel_data]
            )
            if failure_idx is None:
                failure_idx = len(accel_data) - 1
        else:
            failure_idx = len(accel_data) - 1

        failure_time = accel_data[failure_idx]['timestamp']

        # Robust baseline: first 25% of data up to failure, min 3, max 20
        baseline_size = max(3, min(20, int((failure_idx + 1) * 0.25)))
        baseline_size = min(baseline_size, failure_idx + 1)

        # Build baseline stats {sensor: {feature: (mean, std)}}
        baseline_stats: Dict[str, Dict[str, tuple]] = {}
        for stype in ['acceleration', 'current', 'audio']:
            pts = sensor_data.get(stype, [])
            if not pts:
                continue
            baseline_stats[stype] = {}
            for feat in ALL_FEATURES:
                vals = [float(p[feat]) for p in pts[:baseline_size]
                        if p.get(feat) is not None]
                if vals:
                    baseline_stats[stype][feat] = (float(np.mean(vals)),
                                                    float(np.std(vals)))
                else:
                    baseline_stats[stype][feat] = (0.0, 1e-9)

        # Get fault-specific weights
        fault_key = (fault_name or '').lower()
        weights = self._FAULT_WEIGHTS.get(fault_key, self._DEFAULT_WEIGHTS)
        _log.info(f"Using {len(weights)} weighted parameters for '{fault_key}'")

        # -----------------------------------------------------------------------
        # Onset detection
        # -----------------------------------------------------------------------
        if fault_category == 'SUDDEN':
            deviation_idx = failure_idx
            _log.info(f"⚡ SUDDEN — deviation_idx=failure_idx={failure_idx}")
        else:
            # CUSUM accumulation over weighted deviation scores
            CUSUM_k = 0.5    # slack: ignore deviations below 0.5 weighted-sigma
            CUSUM_h = 3.0    # decision threshold
            CUSUM_WINDOW = 2  # consecutive above-threshold intervals to confirm

            cusum_pos = 0.0
            consecutive = 0
            deviation_idx = None

            for i in range(baseline_size, failure_idx + 1):
                # Build a flat dict of this interval's features across all sensors
                rec: Dict = {}
                for stype in ['acceleration', 'current', 'audio']:
                    pts = sensor_data.get(stype, [])
                    if i < len(pts):
                        # Embed sensor data as nested dict for weight lookup
                        rec[stype] = {feat: pts[i].get(feat) for feat in ALL_FEATURES}

                score = self._weighted_cusum_deviation_score(
                    rec, baseline_stats, weights, min_z=0.8
                )

                # CUSUM update
                cusum_pos = max(0.0, cusum_pos + score - CUSUM_k)

                if cusum_pos > CUSUM_h:
                    consecutive += 1
                    if consecutive >= CUSUM_WINDOW:
                        deviation_idx = i - CUSUM_WINDOW + 1
                        _log.info(
                            f"📈 CUSUM onset at index {deviation_idx} "
                            f"(score={score:.3f}, cusum={cusum_pos:.3f}, "
                            f"threshold={CUSUM_h})"
                        )
                        break
                else:
                    consecutive = 0

            if deviation_idx is None:
                # Fallback: if CUSUM never triggered, use the last index
                deviation_idx = failure_idx
                _log.warning("⚠️ CUSUM found no onset — using failure_idx as fallback")

        # Start extraction 3 baseline points before deviation (or 14 for Healthy Operation to get all 15 points of a full normal event)
        if fault_name and 'healthy' in fault_name.lower():
            start_idx = max(0, deviation_idx - 14)
        else:
            start_idx = max(0, deviation_idx - 3)
        _log.info(f"Extracting intervals [{start_idx} → {failure_idx}]  "
                  f"({failure_idx - start_idx + 1} data points per sensor)")

        # -----------------------------------------------------------------------
        # Extract trend data from start_idx to failure_idx
        # -----------------------------------------------------------------------
        trends: Dict[str, List[Dict]] = {}
        for sensor_type in ['acceleration', 'current', 'audio']:
            all_raw = sensor_data.get(sensor_type, [])
            if not all_raw:
                continue

            sensor_points = []
            for ai in range(start_idx, failure_idx + 1):
                ref_time = accel_data[ai]['timestamp']
                best = min(all_raw, key=lambda p: abs(p['timestamp'] - ref_time))
                sensor_points.append(best)

            trend_data = []
            n_points = len(sensor_points)
            expected_freqs = _get_expected_frequencies(fault_name)
            for idx, point in enumerate(sensor_points):
                pt_freqs = [
                    point.get('frequency1', 0),
                    point.get('frequency2', 0),
                    point.get('frequency3', 0),
                    point.get('frequency4', 0),
                    point.get('frequency5', 0)
                ]
                match_score = _calculate_fault_frequency_match(pt_freqs, expected_freqs)
                
                pd = {
                    'timestamp':  point['timestamp'],
                    'time_delta': (point['timestamp'] - failure_time) / 1000,
                    'mean':       point.get('mean',     0),
                    'max':        point.get('max',      0),
                    'min':        point.get('min',      0),
                    'std_dev':    point.get('std_dev',  0),
                    'range':      point.get('range',    0),
                    'kurtosis':   point.get('kurtosis', 0),
                    'variance':   point.get('variance', 0),
                    'skewness':   point.get('skewness', 0),
                    'rms':          point.get('rms',          0.0),
                    'peak':         point.get('peak',         0.0),
                    'crest_factor': point.get('crest_factor', 0.0),
                    'load_factor':  point.get('load_factor',  1.0),
                    'fault_frequency_match': match_score,
                    'interval_phase': (
                        'baseline'  if idx < n_points * 0.25 else
                        'onset'     if idx < n_points * 0.50 else
                        'degrading' if idx < n_points * 0.90 else
                        'failure'
                    ),
                    'frequency1': point.get('frequency1', 0),
                    'frequency2': point.get('frequency2', 0),
                    'frequency3': point.get('frequency3', 0),
                    'frequency4': point.get('frequency4', 0),
                    'frequency5': point.get('frequency5', 0),
                    'amplitude1': point.get('amplitude1', 0),
                    'amplitude2': point.get('amplitude2', 0),
                    'amplitude3': point.get('amplitude3', 0),
                    'amplitude4': point.get('amplitude4', 0),
                    'amplitude5': point.get('amplitude5', 0),
                }

                # Forward slope (or backward for last point)
                if idx < len(sensor_points) - 1:
                    nxt = sensor_points[idx + 1]
                    dt = nxt['timestamp'] - point['timestamp']
                    if dt > 0:
                        for feat in ALL_FEATURES:
                            pd[f'{feat}_slope'] = (nxt.get(feat, 0) - point.get(feat, 0)) / (dt / 1000)
                    else:
                        for feat in ALL_FEATURES:
                            pd[f'{feat}_slope'] = 0.0
                else:  # Last point: backward difference from previous point
                    prev = sensor_points[idx - 1] if idx > 0 else point
                    dt = point['timestamp'] - prev['timestamp']
                    if dt > 0:
                        for feat in ALL_FEATURES:
                            pd[f'{feat}_slope'] = (point.get(feat, 0) - prev.get(feat, 0)) / (dt / 1000)
                    else:
                        for feat in ALL_FEATURES:
                            pd[f'{feat}_slope'] = 0.0

                trend_data.append(pd)

            trends[sensor_type] = trend_data

        return trends
    
    def create_event(self, event_name: str, failure_time_iso: str, description: str = "", start_time_iso: Optional[str] = None) -> Dict:
        """
        Create a new event with multi-sensor trend tracking BACKWARDS from failure.
        Extracts trends based on aggregated features for acceleration, current, and audio.
        
        Args:
            event_name: Name of the event (e.g., "Bearing Failure")
            failure_time_iso: ISO format timestamp of failure (e.g., "2025-11-27T12:24:00")
            description: Optional description of the event
        """
        # Parse failure time
        try:
            failure_dt = datetime.datetime.fromisoformat(failure_time_iso)
            failure_time_ms = failure_dt.timestamp() * 1000
        except ValueError as e:
            raise ValueError(f"Invalid failure time format: {e}")
        
        # Prevent fallback query contamination for automated event creation (Flaw 8)
        if start_time_iso is None:
            if description and ("auto" in description.lower() or "completed" in description.lower()):
                raise ValueError(
                    "start_time_iso is required for automated/auto-triggered event creation to prevent historical data contamination."
                )
            else:
                import logging as _logging
                _log = _logging.getLogger(__name__)
                _log.warning("⚠️ start_time_iso not provided for manual/UI event creation. Falling back to global last 30 rows.")

        # Load sensor data (with aggregated features for all sensors)
        sensor_data = self._load_all_sensor_data(start_time_iso=start_time_iso)
        
        if not sensor_data.get('acceleration'):
            raise ValueError("No acceleration data available in database tables (acceleration, current, audio). Check: 1) Database connection, 2) Tables are populated, 3) Data exists from last 24 hours")
        
        # Extract multi-sensor trends
        multi_sensor_trends = self._extract_multi_sensor_trends(sensor_data, failure_time_ms, fault_name=event_name)
        
        print(f"\n[Trend Extraction Results]")
        for sensor_type, trends in multi_sensor_trends.items():
            print(f"  {sensor_type}: {len(trends)} data points extracted")

        # ==================== PROCESS LOCALLY (NO DATABASE) ====================
        # Determine the next fault ID
        local_fault_id = self.get_next_fault_id_local(event_name)
        
        # Compute row statistics
        total_rows_inserted = 0
        rows_per_sensor = {}
        for sensor_type, trend_data in multi_sensor_trends.items():
            rows_per_sensor[sensor_type] = len(trend_data)
            total_rows_inserted += len(trend_data)
            
        print(f"\n[SUCCESS] Event processed locally!")
        print(f"  Fault ID: {local_fault_id}")
        print(f"  Total Rows: {total_rows_inserted}")
        for sensor_type, count in rows_per_sensor.items():
            print(f"    - {sensor_type}: {count} rows")

        # ==================== SAVE LOCAL EVENT FILES ====================
        event_name_safe = event_name.replace(' ', '_').replace('/', '-')
        failure_date_str = failure_dt.strftime('%Y%m%d_%H%M%S')
        event_id = f"{event_name_safe}_{failure_date_str}"
        
        # Create correct fault name directory under self.events_dir
        event_dir = os.path.join(self.events_dir, event_name)
        os.makedirs(event_dir, exist_ok=True)
        
        json_path = os.path.join(event_dir, f"{event_id}.json")
        jsonl_path = os.path.join(event_dir, f"{event_id}.jsonl")
        trend_path = os.path.join(event_dir, f"{event_id}_trend.jsonl")
        
        # Calculate metadata
        accel_trends = multi_sensor_trends.get('acceleration', [])
        time_before_failure = abs(accel_trends[0]['time_delta']) if accel_trends else 0
        
        # Compute stats for each sensor (including the primary feature stats)
        sensor_stats = {}
        event_key = (event_name or '').lower()
        PRIMARY_FEATURE = {
            'motor bearing failure': 'kurtosis',
            'motor winding failure': 'kurtosis',
            'motor electrical fault': 'skewness',
            'motor overheating': 'mean',
            'motor shaft misalignment': 'mean',
            'motor stall': 'mean',
            'motor vibration anomaly': 'std_dev',
            'pump cavitation': 'kurtosis',
            'pump impeller damage': 'std_dev',
            'pump seal leakage': 'std_dev',
            'custom event': 'kurtosis'
        }
        primary = PRIMARY_FEATURE.get(event_key, 'mean')

        for sensor_type, trends in multi_sensor_trends.items():
            if not trends:
                continue
            values = [t['mean'] for t in trends]
            slopes = [t['mean_slope'] for t in trends[:-1]]
            
            primary_values = [t[primary] for t in trends] if primary in trends[0] else []
            primary_slopes = [t[f'{primary}_slope'] for t in trends[:-1]] if f'{primary}_slope' in trends[0] else []
            
            sensor_stats[sensor_type] = {
                'data_points': len(trends),
                'max_value': max(values) if values else 0,
                'min_value': min(values) if values else 0,
                'avg_value': sum(values) / len(values) if values else 0,
                'max_slope': max(slopes) if slopes else 0,
                'min_slope': min(slopes) if slopes else 0,
                'avg_slope': sum(slopes) / len(slopes) if slopes else 0,
                'primary_feature': primary,
                'primary_max_value': max(primary_values) if primary_values else 0,
                'primary_min_value': min(primary_values) if primary_values else 0,
                'primary_avg_value': sum(primary_values) / len(primary_values) if primary_values else 0,
                'primary_max_slope': max(primary_slopes) if primary_slopes else 0,
                'primary_min_slope': min(primary_slopes) if primary_slopes else 0,
                'primary_avg_slope': sum(primary_slopes) / len(primary_slopes) if primary_slopes else 0,
            }

        metadata = {
            'event_id': event_id,
            'event_name': event_name,
            'description': description,
            'failure_time_iso': failure_time_iso,
            'failure_timestamp_ms': failure_time_ms,
            'actual_data_time_iso': datetime.datetime.fromtimestamp(failure_time_ms / 1000).isoformat(),
            'time_before_failure_seconds': time_before_failure,
            'total_data_points_all_sensors': total_rows_inserted,
            'total_data_points': total_rows_inserted,
            'fault_id_in_database': local_fault_id,
            'fault_id': local_fault_id,
            'rows_per_sensor': rows_per_sensor,
            'sensor_statistics': sensor_stats,
            'created_at': datetime.datetime.now().isoformat()
        }
        
        # Save metadata JSON file
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️ Metadata JSON not saved to file: {e}")

        # Save metadata JSONL file (single line)
        try:
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(metadata) + '\n')
        except Exception as e:
            print(f"⚠️ Metadata JSONL not saved to file: {e}")
            
        # Save trend data JSONL file
        try:
            with open(trend_path, 'w', encoding='utf-8') as f:
                for s_type, s_trends in multi_sensor_trends.items():
                    for pt in s_trends:
                        # Construct a unified trend row dict
                        pt_row = {
                            'fault_id': local_fault_id,
                            'sensor_type': s_type,
                            'timestamp': pt['timestamp'],
                            'timestamp_iso': datetime.datetime.fromtimestamp(pt['timestamp'] / 1000, tz=datetime.timezone.utc).isoformat(),
                            'x_min': pt.get('min', 0.0),
                            'x_max': pt.get('max', 0.0),
                            'mean': pt.get('mean', 0.0),
                            'standard_deviation': pt.get('std_dev', 0.0),
                            'std_dev': pt.get('std_dev', 0.0),
                            'min': pt.get('min', 0.0),
                            'max': pt.get('max', 0.0),
                            'range': pt.get('range', 0.0),
                            'variance': pt.get('variance', 0.0),
                            'skewness': pt.get('skewness', 0.0),
                            'kurtosis': pt.get('kurtosis', 0.0),
                            'mean_slope': pt.get('mean_slope', 0.0),
                            'kurtosis_slope': pt.get('kurtosis_slope', 0.0),
                            'std_dev_slope': pt.get('std_dev_slope', 0.0),
                            'fault_frequency_match': pt.get('fault_frequency_match', 0.0),
                            'interval_phase': pt.get('interval_phase', 'pre_failure'),
                            'rms': pt.get('rms', 0.0),
                            'peak': pt.get('peak', 0.0),
                            'crest_factor': pt.get('crest_factor', 0.0),
                            'load_factor': pt.get('load_factor', 1.0),
                            'rms_slope': pt.get('rms_slope', 0.0),
                            'peak_slope': pt.get('peak_slope', 0.0),
                            'crest_factor_slope': pt.get('crest_factor_slope', 0.0),
                            'frequency1': pt.get('frequency1', 0.0),
                            'frequency2': pt.get('frequency2', 0.0),
                            'frequency3': pt.get('frequency3', 0.0),
                            'frequency4': pt.get('frequency4', 0.0),
                            'frequency5': pt.get('frequency5', 0.0),
                            'amplitude1': pt.get('amplitude1', 0.0),
                            'amplitude2': pt.get('amplitude2', 0.0),
                            'amplitude3': pt.get('amplitude3', 0.0),
                            'amplitude4': pt.get('amplitude4', 0.0),
                            'amplitude5': pt.get('amplitude5', 0.0)
                        }
                        f.write(json.dumps(pt_row) + '\n')
        except Exception as e:
            print(f"⚠️ Trend JSONL not saved to file: {e}")
        
        return {
            'success': True,
            'event_id': event_id,
            'fault_id': local_fault_id,
            'total_rows_inserted': total_rows_inserted,
            'rows_per_sensor': rows_per_sensor,
            'metadata': metadata,
            'multi_sensor_trends': multi_sensor_trends
        }
    
    def list_events(self) -> List[Dict]:
        """
        List all logged events with their metadata.
        Returns empty list if directory doesn't exist (e.g., on Render).
        
        Returns:
            List of event metadata dicts (or empty list if directory doesn't exist)
        """
        events = []
        
        # Handle case where events directory doesn't exist (e.g., on Render)
        if not os.path.exists(self.events_dir):
            return events
        
        try:
            for root, dirs, files in os.walk(self.events_dir):
                for filename in files:
                    if filename.endswith('.json') and filename not in ('stats.json', 'metadata.json'):
                        json_path = os.path.join(root, filename)
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                events.append(metadata)
                        except Exception as e:
                            print(f"Error reading event metadata {filename}: {e}")
                            continue
        except OSError as e:
            print(f"⚠️ Could not list events directory: {e}")
            return events
        
        # Sort by creation time (newest first)
        events.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return events
    
    def get_event(self, event_id: str) -> Optional[Dict]:
        """
        Get detailed data for a specific event locally.

        Reads the metadata JSON (written locally at event creation time) and
        loads trend details from the local trend .jsonl file.

        Args:
            event_id: Event identifier (e.g. "Motor_Stall_20260613_125401")

        Returns:
            Dict with 'metadata' and 'trend_data' (list of per-sensor rows),
            or None if the metadata JSON is not found.
        """
        json_path = None
        # Try finding directly in self.events_dir
        direct_path = os.path.join(self.events_dir, f"{event_id}.json")
        if os.path.exists(direct_path):
            json_path = direct_path
        else:
            # Search in subdirectories
            for root, dirs, files in os.walk(self.events_dir):
                if f"{event_id}.json" in files:
                    json_path = os.path.join(root, f"{event_id}.json")
                    break
                    
        if not json_path:
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            return {'metadata': None, 'trend_data': [], 'error': f"Failed to load metadata: {e}"}

        # Trend data is in {event_id}_trend.jsonl in the same folder as the json file
        parent_dir = os.path.dirname(json_path)
        trend_path = os.path.join(parent_dir, f"{event_id}_trend.jsonl")
        
        trend_data = []
        if os.path.exists(trend_path):
            try:
                with open(trend_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            trend_data.append(json.loads(line))
                # Sort by sensor_type, timestamp
                trend_data.sort(key=lambda x: (x.get('sensor_type', ''), x.get('timestamp', 0)))
            except Exception as e:
                return {'metadata': metadata, 'trend_data': [], 'error': f"Failed to load trend data: {e}"}
                
        return {'metadata': metadata, 'trend_data': trend_data}
    
    def get_unique_event_names(self) -> List[str]:
        """
        Get list of unique event names for dropdown.
        
        Returns:
            List of unique event names
        """
        events = self.list_events()
        unique_names = list(set([event['event_name'] for event in events]))
        unique_names.sort()
        return unique_names
