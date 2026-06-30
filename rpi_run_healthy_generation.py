#!/usr/bin/env python3
"""
RPi Healthy Data Sender
=======================
Generates and uploads normal operating baseline data to the main server.
Compatible with HealthyGenerator physical signal distribution.

Usage:
    python rpi_run_healthy_generation.py --count 500 --interval 5
"""

import sys
import argparse
import logging
import time
import math
import csv
from io import StringIO
from datetime import datetime, timedelta

try:
    import numpy as np
    import requests
except ImportError:
    print("Missing required libraries. Please run:\n  pip install numpy requests")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  –  change SERVER_URL to match the main laptop's IP
# ──────────────────────────────────────────────────────────────────────────────
SERVER_URL = "http://10.145.101.1:5001"
API_KEY    = "sk_prod_7f3b8e2a9c1d4f6e5a2b9c8d7e1f3a5b"

# ──────────────────────────────────────────────────────────────────────────────
# PHYSICS CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
FREQUENCY          = 700            # Hz — samples per second
DURATION           = 2              # seconds per recording window
SAMPLES            = FREQUENCY * DURATION   # 1400 samples
SAMPLE_INTERVAL    = 1000 / FREQUENCY       # ~1.4286 ms between samples

SHAFT_FREQ    = 730 / 60            # ≈ 12.17 Hz
NOM_CURRENT_A = 18.0                # A
NOM_ACCEL_MS2 = 0.5                 # m/s²
NOM_AUDIO_DB  = 88.0                # dB

# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def _generate_time_array():
    return np.linspace(0.0, float(DURATION), SAMPLES, endpoint=False)

def _generate_healthy_acceleration(load_factor):
    t = _generate_time_array()
    N = len(t)
    one_x = NOM_ACCEL_MS2 * load_factor * np.sin(2 * np.pi * SHAFT_FREQ * t)
    noise = np.random.normal(0, NOM_ACCEL_MS2 * load_factor * 0.04, N)
    return one_x + noise

def _generate_healthy_current(load_factor):
    t  = _generate_time_array()
    N  = len(t)
    fla   = NOM_CURRENT_A * load_factor
    noise = np.random.normal(0, fla * 0.03, N)
    return np.clip(fla + noise, 1, 100)

def _generate_healthy_audio(load_factor):
    t    = _generate_time_array()
    N    = len(t)
    base = NOM_AUDIO_DB * load_factor ** 0.05
    shaft_hum = 4.0 * np.sin(2 * np.pi * SHAFT_FREQ * t)
    noise     = np.random.normal(0, 1.5, N)
    return base + shaft_hum + noise

# ──────────────────────────────────────────────────────────────────────────────
# CSV BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def _make_csv_pair(sensor_name, data_array):
    """
    Generate CSV string representing the 1400 samples.
    We return the identical CSV for both max and min files.
    """
    now = datetime.utcnow()
    dt_step = timedelta(seconds=1.0 / FREQUENCY)
    timestamps = [now + i * dt_step for i in range(SAMPLES)]

    buf = StringIO()
    w   = csv.writer(buf)
    w.writerow(['timestamp', 'value'])
    for ts, v in zip(timestamps, data_array):
        fv = float(v)
        if math.isfinite(fv):
            w.writerow([ts.isoformat(), f"{fv:.6f}"])
    csv_str = buf.getvalue()
    return csv_str, csv_str

# ──────────────────────────────────────────────────────────────────────────────
# NETWORK UPLOADER
# ──────────────────────────────────────────────────────────────────────────────

def _upload_sensor(session, sensor_name, max_csv, min_csv, load_factor, logger):
    url     = f"{SERVER_URL.rstrip('/')}/api/upload"
    headers = {
        'X-API-Key': API_KEY,
        'X-Load-Factor': f"{load_factor:.6f}"
    }
    files   = [
        ('files', (f'max_{sensor_name}.csv', max_csv.encode('utf-8'), 'text/csv')),
        ('files', (f'min_{sensor_name}.csv', min_csv.encode('utf-8'), 'text/csv')),
    ]
    for attempt in range(3):
        try:
            resp = session.post(url, files=files, headers=headers, timeout=15)
            if resp.status_code in (200, 201):
                return True
            logger.warning(f"   ⚠ {sensor_name} attempt {attempt+1}: HTTP {resp.status_code}")
        except Exception as exc:
            logger.warning(f"   ⚠ {sensor_name} attempt {attempt+1}: {exc}")
        time.sleep(2)
    return False

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Remote healthy data sender.")
    parser.add_argument("--count",    type=int, default=500, help="Number of intervals to send")
    parser.add_argument("--interval", type=int, default=5,   help="Sleep seconds between intervals")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [RpiSender] - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("RpiSender")

    session = requests.Session()

    # Verify server reachability
    try:
        r = session.get(f"{SERVER_URL.rstrip('/')}/health", timeout=5)
        if r.status_code == 200:
            logger.info("✅ Server connection successful!")
        else:
            logger.warning(f"Server returned {r.status_code} on /health")
    except Exception as exc:
        logger.error(f"❌ Cannot reach server at {SERVER_URL}")
        logger.error(f"   Error: {exc}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🚀 RPi Healthy Data Sender (Cleaned)")
    logger.info(f"   Server   : {SERVER_URL}")
    logger.info(f"   Intervals: {args.count}  |  Sleep: {args.interval}s")
    logger.info("=" * 60)

    success_count = 0
    try:
        for i in range(1, args.count + 1):
            # Roll random load factor matching run_healthy_generation.py
            load_factor = float(np.random.uniform(0.6, 1.1))

            logger.info(f"🔄 Interval {i}/{args.count} (load_factor={load_factor:.3f})")

            # Generate signals using identical physics formulas
            accel_arr = _generate_healthy_acceleration(load_factor)
            curr_arr  = _generate_healthy_current(load_factor)
            audio_arr = _generate_healthy_audio(load_factor)

            sensors = {
                'acceleration': accel_arr,
                'current':      curr_arr,
                'audio':        audio_arr,
            }

            batch_ok = True
            for sname, sarr in sensors.items():
                max_csv, min_csv = _make_csv_pair(sname, sarr)
                ok = _upload_sensor(session, sname, max_csv, min_csv, load_factor, logger)
                if ok:
                    logger.info(f"   ✓ {sname.capitalize()} uploaded")
                else:
                    logger.error(f"   ❌ {sname.capitalize()} FAILED")
                    batch_ok = False

            if batch_ok:
                success_count += 1

            if i < args.count:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")

    logger.info("=" * 60)
    logger.info("🏁 Remote Generation Completed")
    logger.info(f"   Sent successfully: {success_count} / {args.count} intervals")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
