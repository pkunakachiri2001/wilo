from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import sys

# Dynamically add WILO-CLOUD-MONITORING-server subdirectory to path
_base_dir = os.path.dirname(os.path.abspath(__file__))
_server_dir = os.path.join(_base_dir, 'WILO-CLOUD-MONITORING-server')
if os.path.exists(_server_dir) and _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

import glob
import csv
import shutil
import json
import io
import numpy as np
from scipy import stats
import datetime as dt
import logging
import time
import threading
from dotenv import load_dotenv
from event_manager import EventManager
from fft_analysis import calculate_fft_analysis, calculate_fft_full_spectrum
import anomaly_detector
import fault_classifier

load_dotenv()

# ── Static folder ──────────────────────────────────────────────────────────────
static_folder = 'frontend/dist'
if not os.path.exists(os.path.join(_base_dir, static_folder)):
    nested_path = 'WILO-CLOUD-MONITORING-server/frontend/dist'
    if os.path.exists(os.path.join(_base_dir, nested_path)):
        static_folder = nested_path

app = Flask(__name__, static_folder=static_folder, static_url_path='')

CORS(app,
     origins="*",
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
     allow_headers=['Content-Type', 'Authorization', 'X-API-Key'],
     supports_credentials=False,
     max_age=3600)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Directory configuration ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'Data')
EVENTS_DIR = os.path.join(BASE_DIR, 'Events')
UPLOAD_LOG_DIR = os.path.join(BASE_DIR, 'UploadLogs')

for d in [DATA_DIR, EVENTS_DIR, UPLOAD_LOG_DIR]:
    try:
        os.makedirs(d, exist_ok=True)
    except PermissionError as e:
        logger.warning(f"Cannot create {d}: {e}")

logger.info(f"📍 Running locally — Data: {DATA_DIR}, Events: {EVENTS_DIR}")

event_manager = EventManager(EVENTS_DIR, DATA_DIR)

# ── Sensor / upload config ─────────────────────────────────────────────────────
SENSORS = ['acceleration', 'current', 'audio']
SAMPLING_RATE = 700
MAX_FILE_SIZE  = 10 * 1024 * 1024
MAX_CSV_ROWS   = 10000
UPLOAD_API_KEYS = {
    'sensor-001': 'sk_prod_7f3b8e2a9c1d4f6e5a2b9c8d7e1f3a5b',
    'sensor-002': 'sk_prod_2c5d8f1a4e7b9a3d6f2e5c8b1a4d7f3e'
}
UPLOAD_BATCH_SIZE = 2
UPLOAD_FREQUENCY_MINUTES = 110

_batch_lock  = threading.Lock()
_batch_state = {}
_BATCH_WINDOW = 15

# ──────────────────────────────────────────────────────────────────────────────
# FILE READING HELPERS  (with mtime-based cache + race-condition protection)
# ──────────────────────────────────────────────────────────────────────────────
_file_cache: dict = {}         # {filepath: {'mtime': float, 'data': list}}
_dp_cache:   dict = {}         # {(filepath, batch_type): {'mtime': float, 'data': dict|None}}
_cache_lock = threading.Lock()

HIST_LIMIT = 50    # maximum historical entries returned to frontend
TAIL_CHUNK = 65536 # bytes read per chunk during tail-read (64 KB)


def _read_jsonl_safe(filepath: str) -> list:
    """Read every valid JSON line from a JSONL file.
    
    - Skips blank lines.
    - Catches and skips broken (partial-write) last lines.
    Returns a list of dicts, ordered as stored (oldest→newest).
    """
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    lines.append(json.loads(raw))
                except json.JSONDecodeError:
                    # Skip broken last line (partial write race condition)
                    continue
    except OSError as exc:
        logger.warning(f"Cannot open {filepath}: {exc}")
    return lines


def read_jsonl_cached(filepath: str) -> list:
    """Return parsed JSONL content, served from an mtime-keyed in-memory cache.
    
    Cache is invalidated automatically when the file is modified.
    """
    if not os.path.exists(filepath):
        return []
    try:
        mtime = os.path.getmtime(filepath)
    except OSError:
        return _read_jsonl_safe(filepath)

    with _cache_lock:
        entry = _file_cache.get(filepath)
        if entry and entry['mtime'] == mtime:
            return entry['data']

    # Cache miss — read from disk outside the lock to avoid blocking
    data = _read_jsonl_safe(filepath)

    with _cache_lock:
        _file_cache[filepath] = {'mtime': mtime, 'data': data}

    return data


def tail_read_jsonl(filepath: str, n: int) -> list:
    """Efficiently read the last *n* complete JSON lines from a large JSONL file.

    Uses backwards binary seek so we never load the whole file into memory.
    Skips blank lines and silently drops broken (partial-write) lines.
    Returns a list ordered oldest→newest.
    """
    if not os.path.exists(filepath):
        return []

    results = []
    try:
        with open(filepath, 'rb') as fh:
            fh.seek(0, 2)
            pos = fh.tell()
            buf = b''

            while pos > 0 and len(results) < n:
                read_sz = min(TAIL_CHUNK, pos)
                pos -= read_sz
                fh.seek(pos)
                chunk = fh.read(read_sz)
                buf = chunk + buf

                parts = buf.split(b'\n')
                # Keep the first fragment (may be incomplete) for next iteration
                buf = parts[0]

                for part in reversed(parts[1:]):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        results.append(json.loads(part))
                        if len(results) >= n:
                            break
                    except json.JSONDecodeError:
                        continue   # Skip broken lines

    except OSError as exc:
        logger.warning(f"tail_read error on {filepath}: {exc}")

    return list(reversed(results))


def get_last_datapoint_for_mode(sensor: str, batch_type: str) -> dict | None:
    """Return the most recent datapoints entry matching *batch_type*.

    Results are cached per (filepath, batch_type) and keyed by file mtime.
    After the first cold read the function is a near-zero-cost dict lookup;
    the cache is only invalidated when a new upload changes the file.
    """
    filepath = os.path.join(DATA_DIR, f'{sensor}_datapoints.jsonl')
    if not os.path.exists(filepath):
        return None

    try:
        mtime = os.path.getmtime(filepath)
    except OSError:
        mtime = None

    cache_key = (filepath, batch_type)
    if mtime is not None:
        with _cache_lock:
            entry = _dp_cache.get(cache_key)
            if entry and entry['mtime'] == mtime:
                return entry['data']

    # Cache miss — tail-read the file (20 lines; each line ~45 KB)
    candidates = tail_read_jsonl(filepath, 20)
    result = None
    for entry in reversed(candidates):
        if entry.get('batch') == batch_type:
            result = entry
            break

    if mtime is not None:
        with _cache_lock:
            _dp_cache[cache_key] = {'mtime': mtime, 'data': result}

    return result


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS / FFT HELPERS  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_statistics(values, load_factor=1.0):
    if not values:
        return {}

    def safe_float(v):
        f = float(v)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else f

    z = np.array(values)
    rms_val = safe_float(np.sqrt(np.mean(z**2)))
    peak_val = safe_float(np.max(np.abs(z)))
    crest_factor = peak_val / rms_val if rms_val > 1e-9 else 0.0

    return {
        'mean':     safe_float(np.mean(z)),
        'max':      safe_float(np.max(z)),
        'min':      safe_float(np.min(z)),
        'std_dev':  safe_float(np.std(z)),
        'range':    safe_float(np.max(z) - np.min(z)),
        'skewness': safe_float(stats.skew(z)),
        'kurtosis': safe_float(stats.kurtosis(z)),
        'rms':      rms_val,
        'peak':     peak_val,
        'crest_factor': crest_factor,
        'load_factor': float(load_factor),
    }


def get_sensor_health_status(stats_dict):
    if not stats_dict:
        return 'unknown'
    kurtosis  = stats_dict.get('kurtosis', 0)
    std_dev   = stats_dict.get('std_dev', 0)
    if kurtosis > 5.0:
        return 'critical'
    elif kurtosis > 3.0 or std_dev > 2.0:
        return 'warning'
    return 'normal'


def _row_to_stats(row: dict) -> dict:
    """Map a JSONL stats row to the frontend stats shape."""
    return {
        'min':      float(row.get('x_min', 0) or 0),
        'max':      float(row.get('x_max', 0) or 0),
        'mean':     float(row.get('mean', 0) or 0),
        'std_dev':  float(row.get('standard_deviation', 0) or 0),
        'range':    float(row.get('range', 0) or 0),
        'skewness': float(row.get('skewness', 0) or 0),
        'kurtosis': float(row.get('kurtosis', 0) or 0),
        'rms':          float(row.get('rms', 0) or 0),
        'peak':         float(row.get('peak', 0) or 0),
        'crest_factor': float(row.get('crest_factor', 0) or 0),
        'load_factor':  float(row.get('load_factor', 1) or 1),
    }


def _row_to_frequencies(row: dict) -> list:
    return [float(row.get(f'frequency{i}', 0) or 0) for i in range(1, 6)]


def _row_to_amplitudes(row: dict) -> list:
    return [float(row.get(f'amplitude{i}', 0) or 0) for i in range(1, 6)]


def _format_hist_row(row: dict) -> dict:
    """Convert a raw JSONL stats row into the historicalStats shape expected
    by StatisticalAnalysisChart in App.jsx."""
    return {
        'file_timestamp': row.get('created_at'),
        'stats':       _row_to_stats(row),
        'frequencies': _row_to_frequencies(row),
        'amplitudes':  _row_to_amplitudes(row),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CSV helpers  (kept for upload / local CSV fallback — unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_data(filename):
    """Load CSV data and return (timestamps_ms, values, file_modified_iso)."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return [], [], None

    timestamps, values = [], []
    file_modified_iso = None
    try:
        try:
            mtime = os.path.getmtime(filepath)
            file_modified_iso = dt.datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            pass

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_str = row.get('timestamp', '')
                    if isinstance(ts_str, str) and 'T' in ts_str:
                        ts = dt.datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp() * 1000
                    else:
                        ts_val = float(ts_str)
                        ts = ts_val * 1000 if ts_val < 1e11 else ts_val
                    timestamps.append(ts)
                    values.append(float(row.get('value', 0)))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")

    return timestamps, values, file_modified_iso


def merge_max_min_files(max_ts, max_vals, min_ts, min_vals):
    if not max_vals or not min_vals:
        return [], []
    combined = sorted(
        [(ts, v) for ts, v in zip(max_ts, max_vals)] +
        [(ts, v) for ts, v in zip(min_ts, min_vals)],
        key=lambda x: x[0]
    )
    return [c[0] for c in combined], [c[1] for c in combined]


def process_csv_from_memory(file_obj, filename):
    timestamps, values = [], []
    try:
        file_obj.seek(0)
        content = file_obj.read().decode('utf-8')
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return [], [], 0
        header = lines[0].split(',')
        ts_idx = val_idx = None
        for i, col in enumerate(header):
            cl = col.strip().lower()
            if cl == 'timestamp': ts_idx = i
            elif cl == 'value':   val_idx = i
        if ts_idx is None or val_idx is None:
            return [], [], 0
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) <= max(ts_idx, val_idx):
                continue
            try:
                ts_str = parts[ts_idx].strip()
                if 'T' in ts_str:
                    ts = dt.datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp() * 1000
                else:
                    ts_val = float(ts_str)
                    ts = ts_val * 1000 if ts_val < 1e11 else ts_val
                timestamps.append(ts)
                values.append(float(parts[val_idx].strip()))
            except (ValueError, IndexError):
                continue
    except Exception as e:
        logger.error(f"Error processing CSV {filename}: {e}")
    return timestamps, values, len(values)


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST LOGGING
# ──────────────────────────────────────────────────────────────────────────────

@app.before_request
def log_request():
    logger.info(f"REQUEST: {request.method} {request.path}")


@app.after_request
def log_response(response):
    logger.info(f"RESPONSE: {response.status}")
    return response


# ──────────────────────────────────────────────────────────────────────────────
# UPLOAD helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_filename(filename):
    valid = [f'{ft}_{s}.csv' for ft in ['max', 'min'] for s in SENSORS]
    return filename in valid


def validate_csv_file(file):
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        if not content.strip():
            return {'valid': False, 'error': 'Empty file'}
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return {'valid': False, 'error': 'No data rows'}
        header = [c.strip() for c in lines[0].split(',')]
        if 'timestamp' not in header or 'value' not in header:
            return {'valid': False, 'error': 'Missing columns: timestamp, value'}
        rc = sum(1 for l in lines[1:] if l.strip())
        if rc == 0:
            return {'valid': False, 'error': 'No valid data rows'}
        if rc > MAX_CSV_ROWS:
            return {'valid': False, 'error': f'Too many rows: {rc}'}
        file.seek(0)
        return {'valid': True, 'row_count': rc}
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def _append_stats_to_jsonl(sensor: str, file_type: str, stat_row: dict) -> None:
    """Append a statistics record to Data/{sensor}.jsonl."""
    filepath = os.path.join(DATA_DIR, f'{sensor}.jsonl')
    stat_row['file_type'] = file_type
    stat_row['created_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        with open(filepath, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(stat_row) + '\n')
        # Invalidate cache so next request sees fresh data
        with _cache_lock:
            _file_cache.pop(filepath, None)
    except Exception as e:
        logger.error(f"Failed to append stats to {filepath}: {e}")


def _append_datapoints_to_jsonl(sensor: str, batch_type: str,
                                 timestamps: list, values: list,
                                 batch_timestamp: str) -> None:
    """Append a raw-datapoints record to Data/{sensor}_datapoints.jsonl."""
    filepath = os.path.join(DATA_DIR, f'{sensor}_datapoints.jsonl')
    record = {
        'batch':               batch_type,
        'timestamp':           batch_timestamp,
        'datapoints':          values,
        'datapoint_timestamps': timestamps,
    }
    try:
        with open(filepath, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
    except Exception as e:
        logger.error(f"Failed to append datapoints to {filepath}: {e}")


def _process_and_save_locally(uploaded_files, load_factor=1.0) -> dict:
    """Parse CSV uploads, compute stats + FFT, and append to local JSONL files.

    Returns a results dict {mode: {sensor: bool}}.
    """
    results = {m: {s: False for s in SENSORS} for m in ['max', 'min', 'combined']}
    by_mode: dict = {'max': {}, 'min': {}}

    for file_obj in uploaded_files:
        filename = file_obj.filename
        parts    = filename.replace('.csv', '').split('_')
        if len(parts) < 2:
            continue
        mode   = parts[0]
        sensor = '_'.join(parts[1:])
        if mode not in by_mode or sensor not in SENSORS:
            continue
        ts_list, val_list, _ = process_csv_from_memory(file_obj, filename)
        if val_list:
            by_mode[mode][sensor] = {'ts': ts_list, 'vals': val_list}

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    for sensor in SENSORS:
        max_d = by_mode['max'].get(sensor)
        min_d = by_mode['min'].get(sensor)

        if max_d:
            max_stats = calculate_statistics(max_d['vals'], load_factor=load_factor)
            max_freqs, max_amps = calculate_fft_analysis(max_d['vals'])
            row = {
                'x_min': max_stats['min'], 'x_max': max_stats['max'],
                'mean': max_stats['mean'], 'standard_deviation': max_stats['std_dev'],
                'range': max_stats['range'], 'skewness': max_stats['skewness'],
                'kurtosis': max_stats['kurtosis'],
                'rms': max_stats.get('rms', 0.0), 'peak': max_stats.get('peak', 0.0),
                'crest_factor': max_stats.get('crest_factor', 0.0), 'load_factor': max_stats.get('load_factor', 1.0),
                **{f'frequency{i+1}': max_freqs[i] for i in range(5)},
                **{f'amplitude{i+1}': max_amps[i]  for i in range(5)},
            }
            _append_stats_to_jsonl(sensor, 'max', row)
            _append_datapoints_to_jsonl(sensor, 'max', max_d['ts'], max_d['vals'], now_iso)
            results['max'][sensor] = True

        if min_d:
            min_stats = calculate_statistics(min_d['vals'], load_factor=load_factor)
            min_freqs, min_amps = calculate_fft_analysis(min_d['vals'])
            row = {
                'x_min': min_stats['min'], 'x_max': min_stats['max'],
                'mean': min_stats['mean'], 'standard_deviation': min_stats['std_dev'],
                'range': min_stats['range'], 'skewness': min_stats['skewness'],
                'kurtosis': min_stats['kurtosis'],
                'rms': min_stats.get('rms', 0.0), 'peak': min_stats.get('peak', 0.0),
                'crest_factor': min_stats.get('crest_factor', 0.0), 'load_factor': min_stats.get('load_factor', 1.0),
                **{f'frequency{i+1}': min_freqs[i] for i in range(5)},
                **{f'amplitude{i+1}': min_amps[i]  for i in range(5)},
            }
            _append_stats_to_jsonl(sensor, 'min', row)
            _append_datapoints_to_jsonl(sensor, 'min', min_d['ts'], min_d['vals'], now_iso)
            results['min'][sensor] = True

        if max_d and min_d:
            merged_ts, merged_vals = merge_max_min_files(
                max_d['ts'], max_d['vals'], min_d['ts'], min_d['vals'])
            if merged_vals:
                comb_stats = calculate_statistics(merged_vals, load_factor=load_factor)
                comb_freqs, comb_amps = calculate_fft_analysis(merged_vals)
                row = {
                    'x_min': comb_stats['min'], 'x_max': comb_stats['max'],
                    'mean': comb_stats['mean'], 'standard_deviation': comb_stats['std_dev'],
                    'range': comb_stats['range'], 'skewness': comb_stats['skewness'],
                    'kurtosis': comb_stats['kurtosis'],
                    'rms': comb_stats.get('rms', 0.0), 'peak': comb_stats.get('peak', 0.0),
                    'crest_factor': comb_stats.get('crest_factor', 0.0), 'load_factor': comb_stats.get('load_factor', 1.0),
                    **{f'frequency{i+1}': comb_freqs[i] for i in range(5)},
                    **{f'amplitude{i+1}': comb_amps[i]  for i in range(5)},
                }
                _append_stats_to_jsonl(sensor, 'combined', row)
                _append_datapoints_to_jsonl(sensor, 'combined', merged_ts, merged_vals, now_iso)
                results['combined'][sensor] = True

    return results





# ──────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'}), 200


# ── /api/sensor-data ──────────────────────────────────────────────────────────
@app.route('/api/sensor-data')
def get_sensor_data():
    """Return latest stats + raw waveform for all sensors at the requested mode."""
    t0 = time.time()
    mode = request.args.get('mode', 'max').lower()
    if mode not in ('max', 'min', 'combined'):
        return jsonify({'status': 'error', 'message': f'Invalid mode: {mode}'}), 400

    sensor_data = {}
    for sensor in SENSORS:
        stats_path = os.path.join(DATA_DIR, f'{sensor}.jsonl')

        # --- Latest stats row for this mode (use cached full file) ---
        all_rows  = read_jsonl_cached(stats_path)
        filtered  = [r for r in all_rows if r.get('file_type') == mode]
        latest    = filtered[-1] if filtered else None

        if latest:
            s   = _row_to_stats(latest)
            frq = _row_to_frequencies(latest)
            amp = _row_to_amplitudes(latest)
            ts  = latest.get('created_at')
        else:
            s, frq, amp, ts = {}, [], [], None

        # --- Raw waveform (tail-read datapoints file) ---
        dp = get_last_datapoint_for_mode(sensor, mode)
        raw_vals = dp['datapoints']          if dp else []
        raw_ts   = dp['datapoint_timestamps'] if dp else []

        sensor_data[sensor] = {
            'stats':          s,
            'frequencies':    frq,
            'amplitudes':     amp,
            'health':         get_sensor_health_status(s),
            'data_points':    len(raw_vals),
            'raw_timestamps': raw_ts,
            'raw_values':     raw_vals,
            'file_timestamp': ts,
        }

    return jsonify({
        'status':           'success',
        'mode':             mode,
        'data':             sensor_data,
        'timestamp':        dt.datetime.now(dt.timezone.utc).isoformat(),
        'response_time_ms': round((time.time() - t0) * 1000, 2),
    })


# ── /api/historical-stats ─────────────────────────────────────────────────────
@app.route('/api/historical-stats')
def api_historical_stats():
    """Return the last HIST_LIMIT stats entries (oldest→newest) for a sensor.

    Query params:
        sensor      – 'acceleration' | 'current' | 'audio'
        file_type   – 'max' (default) | 'min' | 'combined'
        limit       – ignored (always capped at HIST_LIMIT=50)
    """
    sensor = request.args.get('sensor')
    if not sensor or sensor not in SENSORS:
        return jsonify({'error': f'sensor must be one of {SENSORS}'}), 400

    file_type = request.args.get('file_type', 'max').lower()
    if file_type not in ('max', 'min', 'combined'):
        file_type = 'max'

    stats_path = os.path.join(DATA_DIR, f'{sensor}.jsonl')
    all_rows   = read_jsonl_cached(stats_path)
    filtered   = [r for r in all_rows if r.get('file_type') == file_type]

    # Last HIST_LIMIT entries, oldest→newest order (already in file order)
    recent = filtered[-HIST_LIMIT:]
    formatted = [_format_hist_row(r) for r in recent]

    return jsonify({
        'status': 'success',
        'sensor': sensor,
        'count':  len(formatted),
        'data':   formatted,
    })


# ── /api/combined-dashboard-data ─────────────────────────────────────────────
@app.route('/api/combined-dashboard-data')
def get_combined_dashboard_data():
    """Latest stats (max file_type) for all sensors + recent file list."""
    latest_stats = {}
    for sensor in SENSORS:
        stats_path = os.path.join(DATA_DIR, f'{sensor}.jsonl')
        all_rows   = read_jsonl_cached(stats_path)
        max_rows   = [r for r in all_rows if r.get('file_type') == 'max']
        row        = max_rows[-1] if max_rows else None
        if row:
            latest_stats[sensor] = {
                **_row_to_stats(row),
                'frequency1': float(row.get('frequency1', 0) or 0),
                'frequency2': float(row.get('frequency2', 0) or 0),
                'frequency3': float(row.get('frequency3', 0) or 0),
                'frequency4': float(row.get('frequency4', 0) or 0),
                'frequency5': float(row.get('frequency5', 0) or 0),
                'amplitude1': float(row.get('amplitude1', 0) or 0),
                'amplitude2': float(row.get('amplitude2', 0) or 0),
                'amplitude3': float(row.get('amplitude3', 0) or 0),
                'amplitude4': float(row.get('amplitude4', 0) or 0),
                'amplitude5': float(row.get('amplitude5', 0) or 0),
                'timestamp':  row.get('created_at'),
                'source':     'local',
            }
        else:
            latest_stats[sensor] = None

    # Recent JSONL files list
    jsonl_files = []
    for sensor in SENSORS:
        for suffix in ['', '_datapoints']:
            fp = os.path.join(DATA_DIR, f'{sensor}{suffix}.jsonl')
            if os.path.exists(fp):
                stat = os.stat(fp)
                jsonl_files.append({
                    'name':     os.path.basename(fp),
                    'sensor':   sensor,
                    'size':     stat.st_size,
                    'modified': dt.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
    jsonl_files.sort(key=lambda x: x['modified'], reverse=True)

    return jsonify({
        'success':        True,
        'database_stats': latest_stats,
        'recent_files':   jsonl_files[:10],
        'timestamp':      dt.datetime.now().isoformat(),
    })


# ── /api/latest-data-timestamp ────────────────────────────────────────────────
@app.route('/api/latest-data-timestamp')
def latest_data_timestamp():
    """Return the most recent created_at across all sensor stats files."""
    max_ts = None
    for sensor in SENSORS:
        fp = os.path.join(DATA_DIR, f'{sensor}.jsonl')
        if not os.path.exists(fp):
            continue
        try:
            # Use mtime as a quick proxy (avoids reading file just for timestamp)
            mtime_iso = dt.datetime.fromtimestamp(
                os.path.getmtime(fp), tz=dt.timezone.utc).isoformat()
            if max_ts is None or mtime_iso > max_ts:
                max_ts = mtime_iso
        except Exception:
            pass

    return jsonify({'status': 'success', 'timestamp': max_ts}), 200


# ── /api/anomaly-score ───────────────────────────────────────────────────────
@app.route('/api/anomaly-score')
def get_anomaly_score():
    """Run Isolation Forest + XGBoost on the latest local data."""
    try:
        # Gather latest max-mode stats for all sensors
        latest = {}
        for sensor in SENSORS:
            stats_path = os.path.join(DATA_DIR, f'{sensor}.jsonl')
            all_rows   = read_jsonl_cached(stats_path)
            max_rows   = [r for r in all_rows if r.get('file_type') == 'max']
            if max_rows:
                latest[sensor] = max_rows[-1]

        if not latest:
            return jsonify({'status': 'model_not_ready'}), 200

        result = anomaly_detector.score_snapshot(latest)
        
        # Integrate XGBoost fault classifier
        if fault_classifier.is_ready():
            clf_res = fault_classifier.predict_fault(latest)
            predicted = clf_res.get('predicted_fault', 'Classifier Error: Prediction Failed')
            confidence = clf_res.get('confidence', 0.0)
            
            # If Isolation Forest detected an anomaly, but classifier says normal operation, it is a mismatch
            if result.get('is_anomaly', False) and predicted == 'normal operation':
                predicted = 'Classification Error: Mismatch (predicted Normal Operation during Anomaly)'
                
            result['predicted_fault'] = predicted
            result['classifier_confidence'] = confidence
            result['fault_distribution'] = clf_res.get('distribution', [])
        else:
            result['predicted_fault'] = 'Classifier Error: Model Not Loaded'
            result['classifier_confidence'] = 0.0
            result['fault_distribution'] = []

        result['timestamp'] = dt.datetime.now(dt.timezone.utc).isoformat()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Anomaly score error: {e}")
        return jsonify({'status': 'model_not_ready', 'error': str(e)}), 200


# ── /api/upload ───────────────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Receive CSV uploads, validate, compute stats, append to local JSONL files."""
    try:
        # Authentication
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key not in UPLOAD_API_KEYS.values():
            return jsonify({'status': 'error', 'message': 'Invalid or missing API key'}), 401

        sensor_id = [k for k, v in UPLOAD_API_KEYS.items() if v == api_key][0]

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        uploaded_files = request.files.getlist('files')
        if len(uploaded_files) != UPLOAD_BATCH_SIZE:
            return jsonify({'error': f'Expected {UPLOAD_BATCH_SIZE} files, got {len(uploaded_files)}'}), 400

        validation_report = []
        upload_timestamp  = dt.datetime.now().isoformat()

        for file in uploaded_files:
            if not file or not file.filename.endswith('.csv'):
                return jsonify({'error': f'Invalid file: {file.filename}'}), 400
            if not validate_filename(file.filename):
                return jsonify({'error': f'Invalid filename: {file.filename}'}), 400

            file.seek(0, os.SEEK_END)
            fsize = file.tell()
            file.seek(0)
            if fsize > MAX_FILE_SIZE:
                return jsonify({'error': f'File too large: {file.filename}'}), 413

            vr = validate_csv_file(file)
            if not vr['valid']:
                return jsonify({'error': f'Invalid CSV: {vr["error"]}'}), 400

            validation_report.append({
                'file': file.filename, 'rows': vr['row_count'],
                'size_kb': round(fsize / 1024, 2), 'status': 'success'
            })

        # Process and save locally
        try:
            lf_header = request.headers.get('X-Load-Factor', '').strip()
            load_factor = float(lf_header) if lf_header else 1.0
            process_results = _process_and_save_locally(uploaded_files, load_factor=load_factor)
            saved_count = sum(
                1 for m in process_results.values()
                for ok in m.values() if ok
            )
        except Exception as e:
            logger.error(f"Local save failed: {e}")
            return jsonify({'error': f'Failed to save locally: {e}'}), 500



        return jsonify({
            'status':   'success',
            'message':  f'Processed {len(uploaded_files)} file(s) and saved locally',
            'sensor_id': sensor_id,
            'files':    [f.filename for f in uploaded_files],
            'timestamp': upload_timestamp,
            'validation_report':   validation_report,
            'local_records_saved': saved_count,
            'next_expected_upload': (dt.datetime.now() + dt.timedelta(minutes=120)).isoformat(),
        }), 201

    except Exception as e:
        logger.error(f'Upload error: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── /api/upload-rpi-file ───────────────────────────────────────────────────────
@app.route('/api/upload-rpi-file', methods=['POST'])
def upload_rpi_file():
    """Receive a single CSV file from the RPi (containing Time (s) and Az (m/s2) columns).
    
    Processes the file to extract statistics and FFT, then saves to Data/acceleration.jsonl
    and Data/acceleration_datapoints.jsonl so the dashboard can display it immediately.
    """
    try:
        filename = request.args.get('filename')
        if not filename:
            return jsonify({'error': 'Missing filename parameter'}), 400

        clean_name = os.path.basename(filename)

        # 1. Read the raw body (CSV data)
        raw_data = request.get_data()
        if not raw_data:
            return jsonify({'error': 'Empty file data'}), 400

        # Parse the CSV from memory
        content = raw_data.decode('utf-8', errors='ignore')
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return jsonify({'error': 'No data rows in CSV'}), 400

        header = lines[0].split(',')
        time_idx = val_idx = None
        for i, col in enumerate(header):
            cl = col.strip().lower()
            # Support standard format (timestamp/value) and ESP32 format (Time (s)/Az (m/s2))
            if cl in ('time (s)', 'timestamp'):
                time_idx = i
            elif cl in ('az (m/s2)', 'value'):
                val_idx = i

        if time_idx is None or val_idx is None:
            return jsonify({'error': f'Header missing time or acceleration columns. Found: {header}'}), 400

        timestamps = []
        values = []

        # Convert Time (s) to absolute milliseconds timestamps for chart compatibility.
        # The ESP32 provides relative seconds (e.g. 0.0000, 0.0100...).
        now_ts = time.time() * 1000  # current time in ms

        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) <= max(time_idx, val_idx):
                continue
            try:
                t_sec = float(parts[time_idx].strip())
                val = float(parts[val_idx].strip())
                timestamps.append(now_ts + (t_sec * 1000))
                values.append(val)
            except (ValueError, IndexError):
                continue

        if not values:
            return jsonify({'error': 'No valid data points parsed'}), 400

        # 2. Compute statistics and FFT for acceleration
        load_factor = 1.0  # default
        stats_val = calculate_statistics(values, load_factor=load_factor)
        freqs, amps = calculate_fft_analysis(values)

        row = {
            'x_min': stats_val['min'],
            'x_max': stats_val['max'],
            'mean': stats_val['mean'],
            'standard_deviation': stats_val['std_dev'],
            'range': stats_val['range'],
            'skewness': stats_val['skewness'],
            'kurtosis': stats_val['kurtosis'],
            'rms': stats_val.get('rms', 0.0),
            'peak': stats_val.get('peak', 0.0),
            'crest_factor': stats_val.get('crest_factor', 0.0),
            'load_factor': stats_val.get('load_factor', 1.0),
            **{f'frequency{i+1}': freqs[i] for i in range(5)},
            **{f'amplitude{i+1}': amps[i] for i in range(5)},
        }

        # Save to acceleration JSONL database for both 'max', 'min', and 'combined' modes
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()

        for mode in ['max', 'min', 'combined']:
            _append_stats_to_jsonl('acceleration', mode, row.copy())
            _append_datapoints_to_jsonl('acceleration', mode, timestamps, values, now_iso)

        # Log confirmation in upload history
        log_file = os.path.join(UPLOAD_LOG_DIR, 'upload_history.log')
        with open(log_file, 'a') as f:
            log_record = {
                'timestamp': now_iso,
                'filename': clean_name,
                'sensor': 'acceleration',
                'rows': len(values),
                'status': 'success'
            }
            f.write(repr(log_record) + '\n')

        logger.info(f"✓ Processed and saved RPi file: {clean_name} ({len(values)} points)")
        return jsonify({'status': 'success', 'message': f'Processed RPi file {clean_name}'}), 201

    except Exception as e:
        logger.error(f"RPi file processing failed: {e}")
        return jsonify({'error': str(e)}), 500


# ── /events ───────────────────────────────────────────────────────────────────
@app.route('/events')
def list_events():
    """List all locally stored fault events from the Events/ folder."""
    events = []
    try:
        for fault_dir in os.listdir(EVENTS_DIR):
            fd_path = os.path.join(EVENTS_DIR, fault_dir)
            if not os.path.isdir(fd_path):
                continue
            for fname in os.listdir(fd_path):
                # Skip stats.json, metadata.json, and trend files
                if fname in ('stats.json', 'metadata.json') or fname.endswith('_trend.jsonl'):
                    continue
                
                fpath = os.path.join(fd_path, fname)
                if fname.endswith('.jsonl'):
                    try:
                        rows = read_jsonl_cached(fpath)
                        for row in rows:
                            events.append(row)
                    except Exception as e:
                        logger.warning(f"Cannot read jsonl {fpath}: {e}")
                elif fname.endswith('.json'):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            row = json.load(f)
                            events.append(row)
                    except Exception as e:
                        logger.warning(f"Cannot read json {fpath}: {e}")
    except Exception as e:
        logger.error(f"Error listing events: {e}")

    # Deduplicate by event_id
    seen_ids = set()
    deduped_events = []
    for evt in events:
        eid = evt.get('event_id')
        if eid:
            if eid not in seen_ids:
                seen_ids.add(eid)
                deduped_events.append(evt)
        else:
            deduped_events.append(evt)
    events = deduped_events

    # Sort newest first
    events.sort(key=lambda e: e.get('created_at', ''), reverse=True)
    return jsonify({'events': events})


# ── /api/create-event-from-history ───────────────────────────────────────────
@app.route('/api/create-event-from-history', methods=['POST'])
def create_event_from_history():
    """Create a fault event by extracting data from local JSONL files."""
    try:
        body       = request.get_json(silent=True) or {}
        fault_name = body.get('fault_name', '').strip()
        if not fault_name:
            return jsonify({'error': 'fault_name is required'}), 400

        from datetime import datetime as _dt, timezone
        failure_time = _dt.now(timezone.utc).isoformat()
        result = event_manager.create_event(fault_name, failure_time, description='')

        if not result.get('success'):
            return jsonify({'error': result.get('error', 'Unknown error')}), 500

        return jsonify({
            'success':    True,
            'fault_name': fault_name,
            'fault_id':   result.get('fault_id'),
            'rows_inserted': result.get('total_rows_inserted', 0),
            'intervals_extracted': result.get('rows_per_sensor', {}),
        }), 201

    except Exception as e:
        logger.error(f"create-event-from-history error: {e}")
        return jsonify({'error': str(e)}), 500


# ── /api/upload/status ───────────────────────────────────────────────────────
@app.route('/api/upload/status')
def upload_status():
    try:
        log_file = os.path.join(UPLOAD_LOG_DIR, 'upload_history.log')
        history  = []
        if os.path.exists(log_file):
            with open(log_file) as f:
                for line in f.readlines()[-50:]:
                    try:
                        history.append(eval(line.strip()))
                    except Exception:
                        pass
        return jsonify({'status': 'success', 'total_uploads': len(history),
                        'recent_uploads': history[-10:]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── /api/files ───────────────────────────────────────────────────────────────
@app.route('/api/files')
def get_files():
    files = []
    for fp in glob.glob(os.path.join(DATA_DIR, '*.jsonl')):
        st = os.stat(fp)
        files.append({'name': os.path.basename(fp), 'size': st.st_size,
                      'modified': dt.datetime.fromtimestamp(st.st_mtime).isoformat()})
    return jsonify(files)


# ── /api/storage-info ────────────────────────────────────────────────────────
@app.route('/api/storage-info')
def storage_info():
    return jsonify({
        'storage_config': {
            'data_directory':   DATA_DIR,
            'events_directory': EVENTS_DIR,
            'data_dir_exists':  os.path.exists(DATA_DIR),
        }
    })


# ── Static SPA fallback ───────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_spa(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    index = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(index):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({'status': 'API running — frontend not built'}), 200


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("🚀 Starting Wilo Cloud Monitoring (local mode — no database)")
    logger.info(f"   DATA_DIR   : {DATA_DIR}")
    logger.info(f"   EVENTS_DIR : {EVENTS_DIR}")
    app.run(host='0.0.0.0', port=5001, debug=False)
