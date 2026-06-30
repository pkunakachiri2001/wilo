import os
import json
from dotenv import load_dotenv
import logging
import time
from datetime import datetime, timezone
import atexit
import threading

# Force local-only mode to prevent any attempt to connect to the database (Neon)
os.environ['LOCAL_ONLY'] = 'true'

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False
    class DummyOperationalError(Exception):
        pass
    class Psycopg2Mock:
        OperationalError = DummyOperationalError
    psycopg2 = Psycopg2Mock()
    RealDictCursor = None

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL')
NORMAL_OPS_DATABASE_URL = os.getenv('NORMAL_OPS_DATABASE_URL') or os.getenv('DATABASE_URL')


# Thread-safe in-memory buffer to batch-write local data files
_LOCAL_WRITE_BUFFER = {}
_LOCAL_WRITE_LOCK = threading.Lock()


def flush_local_storage():
    """
    Flush all buffered local storage items to their respective JSONL files.
    """
    global _LOCAL_WRITE_BUFFER
    
    with _LOCAL_WRITE_LOCK:
        if not _LOCAL_WRITE_BUFFER:
            return
            
        # Create a copy of the buffer content to write, then clear the buffer
        buffer_copy = {k: list(v) for k, v in _LOCAL_WRITE_BUFFER.items() if v}
        _LOCAL_WRITE_BUFFER.clear()
        
    if not buffer_copy:
        return
        
    try:
        # Resolve data directory relative to database.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        for table_name, rows in buffer_copy.items():
            file_path = os.path.join(data_dir, f"{table_name}.jsonl")
            try:
                # Open in append mode once per table to write all buffered rows
                with open(file_path, 'a', encoding='utf-8') as f:
                    for row in rows:
                        f.write(json.dumps(row) + '\n')
                logger.info(f"✓ Flushed {len(rows)} rows to local storage: {file_path}")
            except Exception as fe:
                logger.error(f"Failed to write buffer to {file_path}: {fe}")
                # Put them back in the buffer to prevent data loss
                with _LOCAL_WRITE_LOCK:
                    current_rows = _LOCAL_WRITE_BUFFER.setdefault(table_name, [])
                    _LOCAL_WRITE_BUFFER[table_name] = rows + current_rows
    except Exception as e:
        logger.error(f"Error during local storage flush: {e}")


# Register flush on process exit
atexit.register(flush_local_storage)


def _save_locally(table_name, row_dict):
    """
    Buffer a row of data represented by row_dict locally in memory,
    to be flushed to data/{table_name}.jsonl.
    """
    try:
        # Ensure timestamp is string serialized
        row_to_save = {}
        for k, v in row_dict.items():
            if isinstance(v, datetime):
                row_to_save[k] = v.isoformat()
            else:
                row_to_save[k] = v
                
        if 'created_at' not in row_to_save and 'timestamp' not in row_to_save:
            row_to_save['created_at'] = datetime.now(timezone.utc).isoformat()
            
        with _LOCAL_WRITE_LOCK:
            _LOCAL_WRITE_BUFFER.setdefault(table_name, []).append(row_to_save)
            total_buffered = sum(len(rows) for rows in _LOCAL_WRITE_BUFFER.values())
            
        # Trigger auto-flush if buffer size reaches or exceeds 500 rows
        if total_buffered >= 500:
            flush_local_storage()
            
        return True
    except Exception as e:
        logger.error(f"Failed to buffer row locally to {table_name}: {e}")
        return False


# ==================== FAILURE TABLE MAPPING ====================
FAILURE_TABLE_MAPPING = {
    'Motor Bearing Failure': 'motor_bearing_failure',
    'Motor Electrical Fault': 'motor_electrical_fault',
    'Motor Overheating': 'motor_overheating',
    'Motor Shaft Misalignment': 'motor_shaft_misalignment',
    'Motor Stall': 'motor_stall',
    'Motor Vibration Anomaly': 'motor_vibration_anomaly',
    'Motor Winding Failure': 'motor_winding_failure',
    'Pump Cavitation': 'pump_cavitation',
    'Pump Impeller Damage': 'pump_impeller_damage',
    'Pump Seal Leakage': 'pump_seal_leakage'
}

class ConnectionWrapper:
    """Wrapper to prevent closing the shared database connection during bulk operations"""
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        # Do not close when reusing connection during bulk events generation
        if os.getenv('REUSE_CONNECTION') == 'true':
            pass
        else:
            self._conn.close()

    def real_close(self):
        """Cleanly close the underlying connection on bulk process exit"""
        self._conn.close()

_GLOBAL_CONN = None

def get_connection():
    """Get database connection with timeout, supporting global reuse when enabled"""
    global _GLOBAL_CONN
    if os.getenv('REUSE_CONNECTION') == 'true':
        try:
            if _GLOBAL_CONN is not None:
                # Test connection viability
                with _GLOBAL_CONN._conn.cursor() as test_cur:
                    test_cur.execute("SELECT 1")
                return _GLOBAL_CONN
        except Exception:
            try:
                _GLOBAL_CONN._conn.close()
            except Exception:
                pass
            _GLOBAL_CONN = None

    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        if os.getenv('REUSE_CONNECTION') == 'true':
            _GLOBAL_CONN = ConnectionWrapper(conn)
            return _GLOBAL_CONN
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error (operational): {e}")
        raise
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

_GLOBAL_NORMAL_OPS_CONN = None

def get_normal_ops_connection():
    """Get database connection for normal operations, supporting global reuse when enabled"""
    global _GLOBAL_NORMAL_OPS_CONN
    if os.getenv('REUSE_CONNECTION') == 'true':
        try:
            if _GLOBAL_NORMAL_OPS_CONN is not None:
                # Test connection viability
                with _GLOBAL_NORMAL_OPS_CONN._conn.cursor() as test_cur:
                    test_cur.execute("SELECT 1")
                return _GLOBAL_NORMAL_OPS_CONN
        except Exception:
            try:
                _GLOBAL_NORMAL_OPS_CONN._conn.close()
            except Exception:
                pass
            _GLOBAL_NORMAL_OPS_CONN = None

    try:
        conn = psycopg2.connect(NORMAL_OPS_DATABASE_URL, connect_timeout=5)
        if os.getenv('REUSE_CONNECTION') == 'true':
            _GLOBAL_NORMAL_OPS_CONN = ConnectionWrapper(conn)
            return _GLOBAL_NORMAL_OPS_CONN
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Normal ops database connection error (operational): {e}")
        raise
    except Exception as e:
        logger.error(f"Normal ops database connection error: {e}")
        raise

def save_statistics(sensor_name, mode, stats_dict, frequencies, amplitudes):
    """
    Save calculated statistics and FFT data to Neon database.
    
    Args:
        sensor_name: 'acceleration', 'current', or 'audio'
        mode: 'max', 'min', or 'combined'
        stats_dict: Dictionary with keys: mean, max, min, std_dev, range, skewness, kurtosis.
                    kurtosis represents EXCESS kurtosis (Fisher definition where Gaussian = 0.0),
                    following Scipy default.
        frequencies: List of top 5 frequencies
        amplitudes: List of top 5 amplitudes
    """
    start_time = time.time()
    
    # Map sensor_name to table name
    table_name = sensor_name.lower()
    
    # Validate table name to prevent SQL injection
    valid_tables = ['acceleration', 'current', 'audio']
    if table_name not in valid_tables:
        raise ValueError(f"Invalid sensor name: {sensor_name}")
    
    # Ensure we have exactly 5 frequencies and amplitudes (pad with 0 if needed)
    freqs = (frequencies + [0] * 5)[:5]
    amps = (amplitudes + [0] * 5)[:5]
    
    # Mirror locally
    row_dict = {
        'created_at': datetime.now(timezone.utc),
        'x_min': stats_dict.get('min', 0.0),
        'x_max': stats_dict.get('max', 0.0),
        'mean': stats_dict.get('mean', 0.0),
        'range': stats_dict.get('range', 0.0),
        'standard_deviation': stats_dict.get('std_dev', 0.0),
        'skewness': stats_dict.get('skewness', 0.0),
        'kurtosis': stats_dict.get('kurtosis', 0.0),
        'rms': stats_dict.get('rms', 0.0),
        'peak': stats_dict.get('peak', 0.0),
        'crest_factor': stats_dict.get('crest_factor', 0.0),
        'load_factor': stats_dict.get('load_factor', 1.0),
        'frequency1': freqs[0], 'frequency2': freqs[1], 'frequency3': freqs[2], 'frequency4': freqs[3], 'frequency5': freqs[4],
        'amplitude1': amps[0], 'amplitude2': amps[1], 'amplitude3': amps[2], 'amplitude4': amps[3], 'amplitude5': amps[4],
        'file_type': mode
    }
    _save_locally(table_name, row_dict)

    if os.getenv('LOCAL_ONLY') == 'true':
        return True

    conn = None
    try:
        conn = get_connection()
        connection_time = time.time() - start_time
        cur = conn.cursor()
        
        query = f"""
            INSERT INTO {table_name} 
            (x_min, x_max, mean, range, standard_deviation, skewness, kurtosis,
             rms, peak, crest_factor, load_factor,
             frequency1, frequency2, frequency3, frequency4, frequency5,
             amplitude1, amplitude2, amplitude3, amplitude4, amplitude5, file_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s)
        """
        
        query_start = time.time()
        cur.execute(query, (
            stats_dict.get('min', 0),           # x_min
            stats_dict.get('max', 0),           # x_max
            stats_dict.get('mean', 0),          # mean
            stats_dict.get('range', 0),         # range ✅ NOW INCLUDED
            stats_dict.get('std_dev', 0),       # standard_deviation
            stats_dict.get('skewness', 0),      # skewness
            stats_dict.get('kurtosis', 0),      # kurtosis
            stats_dict.get('rms', 0.0),          # rms
            stats_dict.get('peak', 0.0),         # peak
            stats_dict.get('crest_factor', 0.0), # crest_factor
            stats_dict.get('load_factor', 1.0),  # load_factor
            freqs[0], freqs[1], freqs[2], freqs[3], freqs[4],  # frequency1-5
            amps[0], amps[1], amps[2], amps[3], amps[4],       # amplitude1-5
            mode                                 # file_type signature (max, min, or combined)
        ))
        query_time = time.time() - query_start
        
        commit_start = time.time()
        conn.commit()
        commit_time = time.time() - commit_start
        
        total_time = time.time() - start_time
        logger.info(
            f"✓ {sensor_name} ({mode}): "
            f"connection={connection_time*1000:.1f}ms, "
            f"query={query_time*1000:.1f}ms, "
            f"commit={commit_time*1000:.1f}ms, "
            f"total={total_time*1000:.1f}ms"
        )
        return True
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error saving {mode} statistics for {sensor_name}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def _read_latest_locally(table_name, filter_fn=None):
    """
    Read the latest row from local storage (jsonl file) matching the filter function.
    Reads the file backwards for high performance.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        file_path = os.path.join(data_dir, f"{table_name}.jsonl")
        
        if not os.path.exists(file_path):
            data_dir_cap = os.path.join(base_dir, 'Data')
            file_path = os.path.join(data_dir_cap, f"{table_name}.jsonl")
            if not os.path.exists(file_path):
                return None
                
        with open(file_path, 'rb') as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
            except OSError:
                return None
                
            block_size = 262144  # 256 KB blocks
            data = b""
            position = size
            
            while position > 0:
                grab = min(block_size, position)
                position -= grab
                f.seek(position, os.SEEK_SET)
                chunk = f.read(grab)
                data = chunk + data
                
                lines = data.split(b'\n')
                
                # Keep the incomplete first line for the next iteration
                if position > 0:
                    data = lines[0]
                    lines_to_check = lines[1:]
                else:
                    lines_to_check = lines
                    
                # Search backwards through lines in this chunk
                for line_bytes in reversed(lines_to_check):
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if filter_fn is None or filter_fn(row):
                            return row
                    except Exception:
                        continue
        return None
    except Exception as e:
        logger.error(f"Error reading local storage for {table_name}: {e}")
        return None

def get_latest_statistics(sensor_name):
    """Retrieve latest statistics from database with local jsonl fallback"""
    table_name = sensor_name.lower()
    
    if os.getenv('LOCAL_ONLY') == 'true':
        return _read_latest_locally(table_name)
        
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Validate table name
        valid_tables = ['acceleration', 'current', 'audio']
        if table_name not in valid_tables:
            raise ValueError(f"Invalid sensor name: {sensor_name}")
        
        query = f"""
            SELECT * FROM {table_name}
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        cur.execute(query)
        result = cur.fetchone()
        
        return dict(result) if result else None
        
    except Exception as e:
        logger.warning(f"Database statistics unavailable for {sensor_name}, falling back to local files: {e}")
        return _read_latest_locally(table_name)
    finally:
        if conn:
            conn.close()

def get_all_latest_statistics_by_mode(mode='max'):
    """
    Fetch the latest statistics from all sensors in the database.
    
    Returns a dict formatted for the frontend API response:
    {
        "sensor_name": {
            "stats": {mean, max, min, std_dev, skewness, kurtosis},
            "frequencies": [f1, f2, f3, f4, f5],
            "amplitudes": [a1, a2, a3, a4, a5],
            "raw_values": [],
            "raw_timestamps": []
        }
    }
    
    Args:
        mode: 'max', 'min', or 'combined' (currently ignored since DB stores latest only)
    
    Returns:
        Dict with sensor data or empty dict if DB unavailable
    """
    result = {}
    sensor_names = ['acceleration', 'current', 'audio']
    
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        for sensor_name in sensor_names:
            try:
                query = f"""
                    SELECT 
                        x_min, x_max, mean, range, standard_deviation, skewness, kurtosis,
                        frequency1, frequency2, frequency3, frequency4, frequency5,
                        amplitude1, amplitude2, amplitude3, amplitude4, amplitude5
                    FROM {sensor_name}
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                
                cur.execute(query)
                row = cur.fetchone()
                
                if row:
                    row_dict = dict(row)
                    result[sensor_name] = {
                        'stats': {
                            'min': row_dict.get('x_min', 0),
                            'max': row_dict.get('x_max', 0),
                            'mean': row_dict.get('mean', 0),
                            'range': row_dict.get('range', 0),
                            'std_dev': row_dict.get('standard_deviation', 0),
                            'skewness': row_dict.get('skewness', 0),
                            'kurtosis': row_dict.get('kurtosis', 0)
                        },
                        'frequencies': [
                            row_dict.get('frequency1', 0),
                            row_dict.get('frequency2', 0),
                            row_dict.get('frequency3', 0),
                            row_dict.get('frequency4', 0),
                            row_dict.get('frequency5', 0)
                        ],
                        'amplitudes': [
                            row_dict.get('amplitude1', 0),
                            row_dict.get('amplitude2', 0),
                            row_dict.get('amplitude3', 0),
                            row_dict.get('amplitude4', 0),
                            row_dict.get('amplitude5', 0)
                        ],
                        'raw_values': [],  # Not available from DB-only approach
                        'raw_timestamps': []  # Not available from DB-only approach
                    }
                else:
                    logger.warning(f"No statistics found in database for {sensor_name}")
                    result[sensor_name] = {
                        'stats': {},
                        'frequencies': [],
                        'amplitudes': [],
                        'raw_values': [],
                        'raw_timestamps': []
                    }
            except Exception as e:
                logger.error(f"Error fetching stats for {sensor_name}: {e}")
                result[sensor_name] = {
                    'stats': {},
                    'frequencies': [],
                    'amplitudes': [],
                    'raw_values': [],
                    'raw_timestamps': []
                }
        
        logger.info(f"✓ Fetched latest statistics from DB for all sensors")
        return result
        
    except Exception as e:
        logger.error(f"Database connection error in get_all_latest_statistics_by_mode: {e}")
        # Graceful fallback: return empty structure so frontend doesn't break
        return {
            'acceleration': {'stats': {}, 'frequencies': [], 'amplitudes': [], 'raw_values': [], 'raw_timestamps': []},
            'current': {'stats': {}, 'frequencies': [], 'amplitudes': [], 'raw_values': [], 'raw_timestamps': []},
            'audio': {'stats': {}, 'frequencies': [], 'amplitudes': [], 'raw_values': [], 'raw_timestamps': []}
        }
    finally:
        if conn:
            conn.close()

def test_connection():
    """Test database connection"""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        logger.info("Database connection successful!")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
    finally:
        if conn:
            conn.close()

# ==================== EVENT DATA INSERTION FUNCTIONS ====================

def create_event_table_if_not_exists(table_name):
    """
    Create a failure-specific event table if it doesn't already exist.
    Called before inserting data to ensure table is ready.
    
    Args:
        table_name: Name of the table (e.g., "pump_seal_leakage")
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Create table if it doesn't exist
        create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {table_name}_id SERIAL PRIMARY KEY,
                fault_id INTEGER NOT NULL,
                sensor_type VARCHAR(20) NOT NULL DEFAULT 'acceleration'
                    CHECK (sensor_type IN ('acceleration', 'current', 'audio')),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                x_min FLOAT,
                x_max FLOAT,
                mean FLOAT,
                standard_deviation FLOAT,
                range FLOAT,
                variance FLOAT,
                skewness FLOAT,
                kurtosis FLOAT,
                mean_slope FLOAT DEFAULT 0.0,
                kurtosis_slope FLOAT DEFAULT 0.0,
                std_dev_slope FLOAT DEFAULT 0.0,
                fault_frequency_match FLOAT DEFAULT 0.0,
                interval_phase VARCHAR(12) DEFAULT 'pre_failure'
                    CHECK (interval_phase IN ('baseline', 'onset', 'degrading', 'failure')),
                rms FLOAT DEFAULT 0.0,
                peak FLOAT DEFAULT 0.0,
                crest_factor FLOAT DEFAULT 0.0,
                load_factor FLOAT DEFAULT 1.0,
                rms_slope FLOAT DEFAULT 0.0,
                peak_slope FLOAT DEFAULT 0.0,
                crest_factor_slope FLOAT DEFAULT 0.0,
                frequency1 FLOAT,
                frequency2 FLOAT,
                frequency3 FLOAT,
                frequency4 FLOAT,
                frequency5 FLOAT,
                amplitude1 FLOAT,
                amplitude2 FLOAT,
                amplitude3 FLOAT,
                amplitude4 FLOAT,
                amplitude5 FLOAT
            );
        """
        
        cur.execute(create_query)
        
        # Migrations for existing event tables
        migrations = [
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS mean_slope FLOAT DEFAULT 0.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS kurtosis_slope FLOAT DEFAULT 0.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS std_dev_slope FLOAT DEFAULT 0.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS fault_frequency_match FLOAT DEFAULT 0.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS interval_phase VARCHAR(12) DEFAULT 'pre_failure';",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS rms FLOAT DEFAULT 0.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS peak FLOAT DEFAULT 0.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS crest_factor FLOAT DEFAULT 0.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS load_factor FLOAT DEFAULT 1.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS rms_slope FLOAT DEFAULT 0.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS peak_slope FLOAT DEFAULT 0.0;",
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS crest_factor_slope FLOAT DEFAULT 0.0;"
        ]
        for migration in migrations:
            try:
                cur.execute(migration)
            except Exception as migration_error:
                logger.warning(f"Migration failed or already applied to {table_name}: {migration_error}")

        # Migrations for existing raw sensor tables
        for sensor_table in ['acceleration', 'current', 'audio']:
            try:
                cur.execute(f"ALTER TABLE {sensor_table} ADD COLUMN IF NOT EXISTS rms FLOAT DEFAULT 0.0;")
                cur.execute(f"ALTER TABLE {sensor_table} ADD COLUMN IF NOT EXISTS peak FLOAT DEFAULT 0.0;")
                cur.execute(f"ALTER TABLE {sensor_table} ADD COLUMN IF NOT EXISTS crest_factor FLOAT DEFAULT 0.0;")
                cur.execute(f"ALTER TABLE {sensor_table} ADD COLUMN IF NOT EXISTS load_factor FLOAT DEFAULT 1.0;")
            except Exception as se:
                logger.warning(f"Failed to migrate sensor table {sensor_table}: {se}")
        
        # Create indices
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_fault_id ON {table_name}(fault_id);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp);")
        
        conn.commit()
        logger.info(f"✓ Event table '{table_name}' and raw sensor tables verified/migrated")
        
    except Exception as e:
        logger.error(f"Error creating event table '{table_name}': {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass


def get_next_fault_id(failure_type, conn=None):
    """
    Get the next fault_id for a given failure type.
    Fault ID increments per event for each failure type independently.
    Supports transaction reuse to avoid race conditions.
    
    Args:
        failure_type: Name of the failure (e.g., "Motor Stall")
        conn: Optional existing database connection
        
    Returns:
        Next fault_id (integer starting from 1)
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        table_name = FAILURE_TABLE_MAPPING.get(failure_type)
        if not table_name:
            raise ValueError(f"Unknown failure type: {failure_type}")
        
        # Ensure table exists before querying
        create_event_table_if_not_exists(table_name)
        
        cur = conn.cursor()
        # Lock the table in exclusive mode to ensure race-free fault_id assignment
        cur.execute(f"LOCK TABLE {table_name} IN EXCLUSIVE MODE")
        
        # Get max fault_id from table
        query = f"SELECT MAX(fault_id) FROM {table_name}"
        cur.execute(query)
        result = cur.fetchone()
        
        max_fault_id = result[0] if result[0] else 0
        next_id = max_fault_id + 1
        
        logger.info(f"✓ Next fault_id for {failure_type}: {next_id}")
        return next_id
        
    except Exception as e:
        logger.error(f"Error getting next fault_id for {failure_type}: {e}")
        raise
    finally:
        if should_close and conn:
            conn.close()


def insert_event_data(failure_type, multi_sensor_trends):
    """
    Insert multi-sensor event data into the appropriate failure table in Neon.
    
    Args:
        failure_type: Name of the failure (e.g., "Motor Stall")
        multi_sensor_trends: Dict with keys 'acceleration', 'current', 'audio'
                            Each contains list of dicts with trend data and aggregated features
    
    Returns:
        Dict with fault_id, total_rows_inserted, and rows_per_sensor
    """
    conn = None
    total_rows_inserted = 0
    rows_per_sensor = {}
    fault_id = None
    
    try:
        # Get table name and validate
        table_name = FAILURE_TABLE_MAPPING.get(failure_type)
        if not table_name:
            raise ValueError(f"Unknown failure type: {failure_type}")

        if os.getenv('LOCAL_ONLY') == 'true':
            total_rows_inserted = 0
            rows_per_sensor = {}
            fault_id = 999  # Dummy fault_id for local-only execution
            
            for sensor_type, trend_data in multi_sensor_trends.items():
                sensor_rows = 0
                for point in trend_data:
                    timestamp_dt = datetime.fromtimestamp(point['timestamp'] / 1000, tz=timezone.utc)
                    row_dict = {
                        'fault_id': fault_id,
                        'sensor_type': sensor_type,
                        'timestamp': timestamp_dt,
                        'x_min': point.get('min', 0.0),
                        'x_max': point.get('max', 0.0),
                        'mean': point.get('mean', 0.0),
                        'standard_deviation': point.get('std_dev', 0.0),
                        'range': point.get('range', 0.0),
                        'variance': point.get('variance', 0.0),
                        'skewness': point.get('skewness', 0.0),
                        'kurtosis': point.get('kurtosis', 0.0),
                        'mean_slope': point.get('mean_slope', 0.0),
                        'kurtosis_slope': point.get('kurtosis_slope', 0.0),
                        'std_dev_slope': point.get('std_dev_slope', 0.0),
                        'fault_frequency_match': point.get('fault_frequency_match', 0.0),
                        'interval_phase': point.get('interval_phase', 'pre_failure'),
                        'rms': point.get('rms', 0.0),
                        'peak': point.get('peak', 0.0),
                        'crest_factor': point.get('crest_factor', 0.0),
                        'load_factor': point.get('load_factor', 1.0),
                        'rms_slope': point.get('rms_slope', 0.0),
                        'peak_slope': point.get('peak_slope', 0.0),
                        'crest_factor_slope': point.get('crest_factor_slope', 0.0),
                        'frequency1': point.get('frequency1', 0.0),
                        'frequency2': point.get('frequency2', 0.0),
                        'frequency3': point.get('frequency3', 0.0),
                        'frequency4': point.get('frequency4', 0.0),
                        'frequency5': point.get('frequency5', 0.0),
                        'amplitude1': point.get('amplitude1', 0.0),
                        'amplitude2': point.get('amplitude2', 0.0),
                        'amplitude3': point.get('amplitude3', 0.0),
                        'amplitude4': point.get('amplitude4', 0.0),
                        'amplitude5': point.get('amplitude5', 0.0)
                    }
                    _save_locally(table_name, row_dict)
                    sensor_rows += 1
                    total_rows_inserted += 1
                rows_per_sensor[sensor_type] = sensor_rows
                
            logger.info(f"✓ [LOCAL-ONLY] Event saved locally to {table_name}.jsonl")
            return {
                'success': True,
                'fault_id': fault_id,
                'total_rows_inserted': total_rows_inserted,
                'rows_per_sensor': rows_per_sensor,
                'table_name': table_name
            }
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Get next fault_id within the same transaction to guarantee atomicity and race safety
        fault_id = get_next_fault_id(failure_type, conn)
        
        # Prepare insert query
        query = f"""
            INSERT INTO {table_name}
            (fault_id, sensor_type, timestamp, x_min, x_max, mean, standard_deviation,
             range, variance, skewness, kurtosis,
             mean_slope, kurtosis_slope, std_dev_slope, fault_frequency_match, interval_phase,
             rms, peak, crest_factor, load_factor,
             rms_slope, peak_slope, crest_factor_slope,
             frequency1, frequency2, frequency3, frequency4, frequency5,
             amplitude1, amplitude2, amplitude3, amplitude4, amplitude5)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Process each sensor's data
        for sensor_type, trend_data in multi_sensor_trends.items():
            sensor_rows = 0
            
            for point in trend_data:
                try:
                    # Convert timestamp (milliseconds) to datetime
                    timestamp_dt = datetime.fromtimestamp(point['timestamp'] / 1000, tz=timezone.utc)
                    
                    # Mirror locally
                    row_dict = {
                        'fault_id': fault_id,
                        'sensor_type': sensor_type,
                        'timestamp': timestamp_dt,
                        'x_min': point.get('min', 0.0),
                        'x_max': point.get('max', 0.0),
                        'mean': point.get('mean', 0.0),
                        'standard_deviation': point.get('std_dev', 0.0),
                        'range': point.get('range', 0.0),
                        'variance': point.get('variance', 0.0),
                        'skewness': point.get('skewness', 0.0),
                        'kurtosis': point.get('kurtosis', 0.0),
                        'mean_slope': point.get('mean_slope', 0.0),
                        'kurtosis_slope': point.get('kurtosis_slope', 0.0),
                        'std_dev_slope': point.get('std_dev_slope', 0.0),
                        'fault_frequency_match': point.get('fault_frequency_match', 0.0),
                        'interval_phase': point.get('interval_phase', 'pre_failure'),
                        'rms': point.get('rms', 0.0),
                        'peak': point.get('peak', 0.0),
                        'crest_factor': point.get('crest_factor', 0.0),
                        'load_factor': point.get('load_factor', 1.0),
                        'rms_slope': point.get('rms_slope', 0.0),
                        'peak_slope': point.get('peak_slope', 0.0),
                        'crest_factor_slope': point.get('crest_factor_slope', 0.0),
                        'frequency1': point.get('frequency1', 0.0),
                        'frequency2': point.get('frequency2', 0.0),
                        'frequency3': point.get('frequency3', 0.0),
                        'frequency4': point.get('frequency4', 0.0),
                        'frequency5': point.get('frequency5', 0.0),
                        'amplitude1': point.get('amplitude1', 0.0),
                        'amplitude2': point.get('amplitude2', 0.0),
                        'amplitude3': point.get('amplitude3', 0.0),
                        'amplitude4': point.get('amplitude4', 0.0),
                        'amplitude5': point.get('amplitude5', 0.0)
                    }
                    _save_locally(table_name, row_dict)
                    
                    cur.execute(query, (
                        fault_id,                              # fault_id
                        sensor_type,                           # sensor_type
                        timestamp_dt,                          # timestamp
                        point.get('min', 0),                   # x_min
                        point.get('max', 0),                   # x_max
                        point.get('mean', 0),                  # mean
                        point.get('std_dev', 0),               # standard_deviation
                        point.get('range', 0),                 # range
                        point.get('variance', 0),              # variance
                        point.get('skewness', 0),              # skewness
                        point.get('kurtosis', 0),              # kurtosis
                        point.get('mean_slope', 0),            # mean_slope
                        point.get('kurtosis_slope', 0),        # kurtosis_slope
                        point.get('std_dev_slope', 0),         # std_dev_slope
                        point.get('fault_frequency_match', 0), # fault_frequency_match
                        point.get('interval_phase', 'pre_failure'), # interval_phase
                        point.get('rms', 0.0),                  # rms
                        point.get('peak', 0.0),                 # peak
                        point.get('crest_factor', 0.0),         # crest_factor
                        point.get('load_factor', 1.0),          # load_factor
                        point.get('rms_slope', 0.0),            # rms_slope
                        point.get('peak_slope', 0.0),           # peak_slope
                        point.get('crest_factor_slope', 0.0),   # crest_factor_slope
                        point.get('frequency1', 0),            # frequency1
                        point.get('frequency2', 0),            # frequency2
                        point.get('frequency3', 0),            # frequency3
                        point.get('frequency4', 0),            # frequency4
                        point.get('frequency5', 0),            # frequency5
                        point.get('amplitude1', 0),            # amplitude1
                        point.get('amplitude2', 0),            # amplitude2
                        point.get('amplitude3', 0),            # amplitude3
                        point.get('amplitude4', 0),            # amplitude4
                        point.get('amplitude5', 0)             # amplitude5
                    ))
                    sensor_rows += 1
                    total_rows_inserted += 1
                    
                except Exception as e:
                    logger.error(f"Error inserting row for {failure_type}/{sensor_type}: {e}")
                    raise
            
            rows_per_sensor[sensor_type] = sensor_rows
        
        # Commit all inserts
        conn.commit()
        
        logger.info(
            f"✓ Multi-sensor event saved to database: "
            f"Table={table_name}, fault_id={fault_id}, total_rows={total_rows_inserted}, "
            f"sensors={list(rows_per_sensor.keys())}"
        )
        
        return {
            'success': True,
            'fault_id': fault_id,
            'total_rows_inserted': total_rows_inserted,
            'rows_per_sensor': rows_per_sensor,
            'table_name': table_name
        }
        
    except Exception as e:
        logger.error(f"Error inserting multi-sensor event data for {failure_type}: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def insert_event_from_historical_data(failure_type, extracted_data):
    """
    Insert event data from historical extraction into the appropriate failure table in Neon.
    Used when creating events from existing historical database records.
    Each data_point dict must contain a 'sensor_type' key ('acceleration', 'current', or 'audio')
    so rows are correctly labelled by sensor.
    
    Args:
        failure_type: Name of the failure (e.g., "Motor Stall")
        extracted_data: List of dicts with historical stat columns + 'sensor_type'
        
    Returns:
        Dict with fault_id, rows_inserted, and table name
    """
    conn = None
    rows_inserted = 0
    fault_id = None
    
    try:
        # Get table name and validate
        table_name = FAILURE_TABLE_MAPPING.get(failure_type)
        if not table_name:
            raise ValueError(f"Unknown failure type: {failure_type}")

        if os.getenv('LOCAL_ONLY') == 'true':
            rows_inserted = 0
            fault_id = 999
            for data_point in extracted_data:
                timestamp_val = data_point.get('timestamp', datetime.now())
                if isinstance(timestamp_val, str):
                    try:
                        timestamp_val = datetime.fromisoformat(timestamp_val)
                    except:
                        timestamp_val = datetime.now()
                sensor_type = data_point.get('sensor_type')
                if sensor_type not in ('acceleration', 'current', 'audio'):
                    raise ValueError(f"Invalid sensor_type: {sensor_type}. Must be 'acceleration', 'current', or 'audio'.")
                    
                row_dict = {
                    'fault_id': fault_id,
                    'sensor_type': sensor_type,
                    'timestamp': timestamp_val,
                    'x_min': data_point.get('min', 0.0),
                    'x_max': data_point.get('max', 0.0),
                    'mean': data_point.get('mean', 0.0),
                    'standard_deviation': data_point.get('std_dev', 0.0),
                    'range': data_point.get('range', 0.0),
                    'variance': data_point.get('variance', 0.0),
                    'skewness': data_point.get('skewness', 0.0),
                    'kurtosis': data_point.get('kurtosis', 0.0),
                    'mean_slope': data_point.get('mean_slope', 0.0),
                    'kurtosis_slope': data_point.get('kurtosis_slope', 0.0),
                    'std_dev_slope': data_point.get('std_dev_slope', 0.0),
                    'fault_frequency_match': data_point.get('fault_frequency_match', 0.0),
                    'interval_phase': data_point.get('interval_phase', 'pre_failure'),
                    'rms': data_point.get('rms', 0.0),
                    'peak': data_point.get('peak', 0.0),
                    'crest_factor': data_point.get('crest_factor', 0.0),
                    'load_factor': data_point.get('load_factor', 1.0),
                    'rms_slope': data_point.get('rms_slope', 0.0),
                    'peak_slope': data_point.get('peak_slope', 0.0),
                    'crest_factor_slope': data_point.get('crest_factor_slope', 0.0),
                    'frequency1': data_point.get('frequency1', 0.0),
                    'frequency2': data_point.get('frequency2', 0.0),
                    'frequency3': data_point.get('frequency3', 0.0),
                    'frequency4': data_point.get('frequency4', 0.0),
                    'frequency5': data_point.get('frequency5', 0.0),
                    'amplitude1': data_point.get('amplitude1', 0.0),
                    'amplitude2': data_point.get('amplitude2', 0.0),
                    'amplitude3': data_point.get('amplitude3', 0.0),
                    'amplitude4': data_point.get('amplitude4', 0.0),
                    'amplitude5': data_point.get('amplitude5', 0.0)
                }
                _save_locally(table_name, row_dict)
                rows_inserted += 1
                
            logger.info(f"✓ [LOCAL-ONLY] Historical event saved locally to {table_name}.jsonl")
            return {
                'success': True,
                'fault_id': fault_id,
                'rows_inserted': rows_inserted,
                'table_name': table_name
            }
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Get next fault_id for this failure type on the active connection
        fault_id = get_next_fault_id(failure_type, conn)
        
        # Prepare insert query - includes sensor_type
        query = f"""
            INSERT INTO {table_name}
            (fault_id, sensor_type, timestamp, x_min, x_max, mean, standard_deviation,
             range, variance, skewness, kurtosis,
             mean_slope, kurtosis_slope, std_dev_slope, fault_frequency_match, interval_phase,
             rms, peak, crest_factor, load_factor,
             rms_slope, peak_slope, crest_factor_slope,
             frequency1, frequency2, frequency3, frequency4, frequency5,
             amplitude1, amplitude2, amplitude3, amplitude4, amplitude5)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Insert each historical data point with the new fault_id
        for data_point in extracted_data:
            try:
                # Parse timestamp if it's a string
                timestamp_val = data_point.get('timestamp', datetime.now())
                if isinstance(timestamp_val, str):
                    try:
                        timestamp_val = datetime.fromisoformat(timestamp_val)
                    except:
                        timestamp_val = datetime.now()
                
                # Determine sensor_type from the data point (tagged in app.py)
                sensor_type = data_point.get('sensor_type')
                if sensor_type not in ('acceleration', 'current', 'audio'):
                    raise ValueError(f"Invalid sensor_type: {sensor_type}. Must be 'acceleration', 'current', or 'audio'.")
                
                # Mirror locally
                row_dict = {
                    'fault_id': fault_id,
                    'sensor_type': sensor_type,
                    'timestamp': timestamp_val,
                    'x_min': data_point.get('min', 0.0),
                    'x_max': data_point.get('max', 0.0),
                    'mean': data_point.get('mean', 0.0),
                    'standard_deviation': data_point.get('std_dev', 0.0),
                    'range': data_point.get('range', 0.0),
                    'variance': data_point.get('variance', 0.0),
                    'skewness': data_point.get('skewness', 0.0),
                    'kurtosis': data_point.get('kurtosis', 0.0),
                    'mean_slope': data_point.get('mean_slope', 0.0),
                    'kurtosis_slope': data_point.get('kurtosis_slope', 0.0),
                    'std_dev_slope': data_point.get('std_dev_slope', 0.0),
                    'fault_frequency_match': data_point.get('fault_frequency_match', 0.0),
                    'interval_phase': data_point.get('interval_phase', 'pre_failure'),
                    'rms': data_point.get('rms', 0.0),
                    'peak': data_point.get('peak', 0.0),
                    'crest_factor': data_point.get('crest_factor', 0.0),
                    'load_factor': data_point.get('load_factor', 1.0),
                    'rms_slope': data_point.get('rms_slope', 0.0),
                    'peak_slope': data_point.get('peak_slope', 0.0),
                    'crest_factor_slope': data_point.get('crest_factor_slope', 0.0),
                    'frequency1': data_point.get('frequency1', 0.0),
                    'frequency2': data_point.get('frequency2', 0.0),
                    'frequency3': data_point.get('frequency3', 0.0),
                    'frequency4': data_point.get('frequency4', 0.0),
                    'frequency5': data_point.get('frequency5', 0.0),
                    'amplitude1': data_point.get('amplitude1', 0.0),
                    'amplitude2': data_point.get('amplitude2', 0.0),
                    'amplitude3': data_point.get('amplitude3', 0.0),
                    'amplitude4': data_point.get('amplitude4', 0.0),
                    'amplitude5': data_point.get('amplitude5', 0.0)
                }
                _save_locally(table_name, row_dict)
                
                cur.execute(query, (
                    fault_id,                                           # fault_id
                    sensor_type,                                        # sensor_type
                    timestamp_val,                                      # timestamp
                    data_point.get('min', 0),                          # x_min
                    data_point.get('max', 0),                          # x_max
                    data_point.get('mean', 0),                         # mean
                    data_point.get('std_dev', 0),                      # standard_deviation
                    data_point.get('range', 0),                        # range
                    data_point.get('variance', 0),                     # variance
                    data_point.get('skewness', 0),                     # skewness
                    data_point.get('kurtosis', 0),                     # kurtosis
                    data_point.get('mean_slope', 0),                   # mean_slope
                    data_point.get('kurtosis_slope', 0),               # kurtosis_slope
                    data_point.get('std_dev_slope', 0),                # std_dev_slope
                    data_point.get('fault_frequency_match', 0),        # fault_frequency_match
                    data_point.get('interval_phase', 'pre_failure'),   # interval_phase
                    data_point.get('rms', 0.0),                         # rms
                    data_point.get('peak', 0.0),                        # peak
                    data_point.get('crest_factor', 0.0),                # crest_factor
                    data_point.get('load_factor', 1.0),                 # load_factor
                    data_point.get('rms_slope', 0.0),                   # rms_slope
                    data_point.get('peak_slope', 0.0),                  # peak_slope
                    data_point.get('crest_factor_slope', 0.0),          # crest_factor_slope
                    data_point.get('frequency1', 0),                   # frequency1
                    data_point.get('frequency2', 0),                   # frequency2
                    data_point.get('frequency3', 0),                   # frequency3
                    data_point.get('frequency4', 0),                   # frequency4
                    data_point.get('frequency5', 0),                   # frequency5
                    data_point.get('amplitude1', 0),                   # amplitude1
                    data_point.get('amplitude2', 0),                   # amplitude2
                    data_point.get('amplitude3', 0),                   # amplitude3
                    data_point.get('amplitude4', 0),                   # amplitude4
                    data_point.get('amplitude5', 0)                    # amplitude5
                ))
                rows_inserted += 1
                
            except Exception as e:
                logger.error(f"Error inserting historical data point for {failure_type}: {e}")
                raise
        
        # Commit all inserts
        conn.commit()
        
        logger.info(
            f"✓ Event saved to database from historical data: "
            f"Table={table_name}, fault_id={fault_id}, rows={rows_inserted}"
        )
        
        return {
            'success': True,
            'fault_id': fault_id,
            'rows_inserted': rows_inserted,
            'table_name': table_name
        }
        
    except Exception as e:
        logger.error(f"Error inserting historical event data for {failure_type}: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


# ==================== RAW DATAPOINTS STORAGE FUNCTIONS ====================

def save_raw_datapoints(sensor_name, batch, timestamp_ms, datapoints, datapoint_timestamps):
    """
    Save raw datapoints and their timestamps to Neon database.
    
    Args:
        sensor_name: 'acceleration', 'current', or 'audio'
        batch: 'max', 'min', or 'combined'
        timestamp_ms: numeric timestamp (ms) for the batch primary timestamp
        datapoints: List or numpy array of float values
        datapoint_timestamps: List or numpy array of bigint timestamps (milliseconds)
    """
    table_name = f"{sensor_name.lower()}_datapoints"
    
    # Validate table name to prevent SQL injection
    valid_tables = ['acceleration_datapoints', 'current_datapoints', 'audio_datapoints']
    if table_name not in valid_tables:
        raise ValueError(f"Invalid sensor table: {table_name}")
        
    # Ensure native Python lists of appropriate types
    if hasattr(datapoints, 'tolist'):
        datapoints_list = datapoints.tolist()
    else:
        datapoints_list = [float(x) for x in datapoints]
        
    if hasattr(datapoint_timestamps, 'tolist'):
        timestamps_list = datapoint_timestamps.tolist()
    else:
        timestamps_list = [int(x) for x in datapoint_timestamps]
        
    # Convert timestamp_ms to datetime object with timezone info
    if timestamp_ms > 1e11:
        timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    else:
        timestamp_dt = datetime.fromtimestamp(timestamp_ms, tz=timezone.utc)
        
    # Mirror locally
    row_dict = {
        'timestamp': timestamp_dt,
        'batch': batch,
        'datapoints': datapoints_list,
        'datapoint_timestamps': timestamps_list
    }
    _save_locally(table_name, row_dict)
    
    if os.getenv('LOCAL_ONLY') == 'true':
        return True

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        query = f"""
            INSERT INTO {table_name} (timestamp, batch, datapoints, datapoint_timestamps)
            VALUES (%s, %s, %s, %s)
        """
        
        cur.execute(query, (timestamp_dt, batch, datapoints_list, timestamps_list))
        conn.commit()
        logger.info(f"✓ Saved raw datapoints for {sensor_name} ({batch})")
        return True
    except Exception as e:
        logger.error(f"Error saving raw datapoints for {sensor_name} ({batch}): {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_latest_raw_datapoints(sensor_name, batch='max'):
    """
    Retrieve the latest raw datapoints for a sensor and batch type.
    """
    table_name = f"{sensor_name.lower()}_datapoints"
    
    def format_local_row(local_row):
        if not local_row:
            return None
        ts = local_row.get('timestamp')
        if hasattr(ts, 'isoformat'):
            ts_str = ts.isoformat()
        else:
            ts_str = ts
        return {
            'timestamp': ts_str,
            'batch': local_row['batch'],
            'datapoints': list(local_row['datapoints']),
            'datapoint_timestamps': [int(t) for t in local_row['datapoint_timestamps']]
        }

    if os.getenv('LOCAL_ONLY') == 'true':
        local_row = _read_latest_locally(table_name, lambda r: r.get('batch') == batch)
        return format_local_row(local_row)

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        valid_tables = ['acceleration_datapoints', 'current_datapoints', 'audio_datapoints']
        if table_name not in valid_tables:
            raise ValueError(f"Invalid sensor table: {table_name}")
            
        query = f"""
            SELECT timestamp, batch, datapoints, datapoint_timestamps
            FROM {table_name}
            WHERE batch = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        cur.execute(query, (batch,))
        row = cur.fetchone()
        
        if row:
            row_dict = dict(row)
            return {
                'timestamp': row_dict['timestamp'].isoformat() if row_dict['timestamp'] else None,
                'batch': row_dict['batch'],
                'datapoints': list(row_dict['datapoints']),
                'datapoint_timestamps': [int(t) for t in row_dict['datapoint_timestamps']]
            }
        return None
    except Exception as e:
        logger.warning(f"Database raw datapoints unavailable for {sensor_name} ({batch}), falling back to local files: {e}")
        local_row = _read_latest_locally(table_name, lambda r: r.get('batch') == batch)
        return format_local_row(local_row)
    finally:
        if conn:
            conn.close()


def create_normal_operations_table_if_not_exists(conn=None):
    """
    Create normal_operations table in Neon if it doesn't already exist.
    Stores clean, normal baseline feature statistics for Isolation Forest training.
    """
    should_close = False
    if conn is None:
        conn = get_normal_ops_connection()
        should_close = True
    try:
        cur = conn.cursor()
        
        # Create normal operations table
        create_query = """
            CREATE TABLE IF NOT EXISTS normal_operations (
                normal_id SERIAL PRIMARY KEY,
                sensor_type VARCHAR(20) NOT NULL CHECK (sensor_type IN ('acceleration', 'current', 'audio')),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                x_min FLOAT,
                x_max FLOAT,
                mean FLOAT,
                standard_deviation FLOAT,
                range FLOAT,
                variance FLOAT,
                skewness FLOAT,
                kurtosis FLOAT,
                rms FLOAT DEFAULT 0.0,
                peak FLOAT DEFAULT 0.0,
                crest_factor FLOAT DEFAULT 0.0,
                load_factor FLOAT DEFAULT 1.0,
                file_type VARCHAR(12) NOT NULL DEFAULT 'max' CHECK (file_type IN ('max', 'min', 'combined')),
                frequency1 FLOAT,
                frequency2 FLOAT,
                frequency3 FLOAT,
                frequency4 FLOAT,
                frequency5 FLOAT,
                amplitude1 FLOAT,
                amplitude2 FLOAT,
                amplitude3 FLOAT,
                amplitude4 FLOAT,
                amplitude5 FLOAT
            );
        """
        cur.execute(create_query)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_normal_operations_sensor_type ON normal_operations(sensor_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_normal_operations_timestamp ON normal_operations(timestamp);")
        
        conn.commit()
        logger.info("✓ normal_operations table verified/created")
        return True
    except Exception as e:
        logger.error(f"Error creating normal_operations table: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if should_close and conn:
            conn.close()


def save_normal_operations_data(sensor_name, mode, stats_dict, frequencies, amplitudes, conn=None):
    """
    Save healthy normal baseline features to the normal_operations table in Neon.
    (Disabled - normal operations file is redundant)
    """
    return True


def get_historical_stats_locally(sensor_name, limit=25):
    """
    Retrieve historical stats locally from JSONL files, matching WHERE file_type = 'max'
    ordered by created_at DESC limit 25.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        file_path = os.path.join(data_dir, f"{sensor_name.lower()}.jsonl")
        
        if not os.path.exists(file_path):
            data_dir_cap = os.path.join(base_dir, 'Data')
            file_path = os.path.join(data_dir_cap, f"{sensor_name.lower()}.jsonl")
            if not os.path.exists(file_path):
                return []
                
        matching_rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row.get('file_type') == 'max':
                        matching_rows.append(row)
                except Exception:
                    continue
        
        # Take the last `limit` elements (which are latest because they are appended at the end)
        latest_rows = matching_rows[-limit:]
        
        # Format rows like the database query does
        formatted_rows = []
        for row in latest_rows:
            # Parse created_at ISO timestamp or format
            created_at_val = row.get('created_at')
            formatted_rows.append({
                'created_at': created_at_val,
                'x_min': row.get('x_min', 0.0),
                'x_max': row.get('x_max', 0.0),
                'mean': row.get('mean', 0.0),
                'standard_deviation': row.get('standard_deviation', 0.0),
                'range': row.get('range', 0.0),
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
                'file_type': row.get('file_type')
            })
        return formatted_rows
    except Exception as e:
        logger.error(f"Error reading local stats for {sensor_name}: {e}")
        return []


