#!/usr/bin/env python3
"""
Bulk Training Data Generator
Generates clean normal baseline statistics and various fault event sequence data.
Both cloud (Neon PostgreSQL) and local storage (JSON Lines) mirror all data.

Usage: python run_bulk_training_data_generation.py --fault-runs 50 --normal-intervals 1000
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "WILO-CLOUD-MONITORING-server"))

# Quiet down chatty loggers to keep output clean and readable
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BulkGenerator")
logger.setLevel(logging.INFO)

# Suppress logs for the generator instances
for name in ['Generator-Motor Stall', 'Generator-Pump Cavitation', 
            'Generator-Pump Impeller Damage', 'Generator-Pump Seal Leakage',
            'Generator-Motor Bearing Failure', 'Generator-Motor Shaft Misalignment',
            'Generator-Motor Overheating', 'Generator-Motor Winding Failure',
            'Generator-Motor Vibration Anomaly', 'Generator-Motor Electrical Fault',
            'Generator-Custom Event', 'Generator-Healthy Operation',
            'database', 'event_manager']:
    logging.getLogger(name).setLevel(logging.WARNING)

# Patch base generator's interval to 0 immediately for high-speed sequential runs
import fault_generators.base_generator as bg
bg.GENERATION_INTERVAL = 0.0

# Import generators
from fault_generators.healthy_generator import HealthyGenerator
from fault_generators.motor_stall_generator import MotorStallGenerator
from fault_generators.pump_cavitation_generator import PumpCavitationGenerator
from fault_generators.pump_impeller_damage_generator import PumpImpellerDamageGenerator
from fault_generators.pump_seal_leakage_generator import PumpSealLeakageGenerator
from fault_generators.motor_bearing_failure_generator import MotorBearingFailureGenerator
from fault_generators.motor_shaft_misalignment_generator import MotorShaftMisalignmentGenerator
from fault_generators.motor_overheating_generator import MotorOverheatingGenerator
from fault_generators.motor_winding_failure_generator import MotorWindingFailureGenerator
from fault_generators.motor_vibration_anomaly_generator import MotorVibrationAnomalyGenerator
from fault_generators.motor_electrical_fault_generator import MotorElectricalFaultGenerator
from fault_generators.custom_event_generator import CustomEventGenerator

FAULT_GENERATORS = [
    ('Motor Stall', MotorStallGenerator),
    ('Pump Cavitation', PumpCavitationGenerator),
    ('Pump Impeller Damage', PumpImpellerDamageGenerator),
    ('Pump Seal Leakage', PumpSealLeakageGenerator),
    ('Motor Bearing Failure', MotorBearingFailureGenerator),
    ('Motor Shaft Misalignment', MotorShaftMisalignmentGenerator),
    ('Motor Overheating', MotorOverheatingGenerator),
    ('Motor Winding Failure', MotorWindingFailureGenerator),
    ('Motor Vibration Anomaly', MotorVibrationAnomalyGenerator),
    ('Motor Electrical Fault', MotorElectricalFaultGenerator),
]


def main():
    parser = argparse.ArgumentParser(description="Bulk Training Data Generator for ML Models (Normal & Fault Data).")
    parser.add_argument(
        "--fault-runs",
        type=int,
        default=50,
        help="Number of times to run each of the 11 fault generators (default: 50)"
    )
    parser.add_argument(
        "--normal-intervals",
        type=int,
        default=250,
        help="Number of healthy normal operation intervals to generate (default: 250)"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Bypass cloud database PostgreSQL inserts and save ONLY to local JSONL files (extreme high-speed)"
    )
    args = parser.parse_args()

    # Enable database connection reuse to avoid TCP handshake overhead on Neon
    os.environ['REUSE_CONNECTION'] = 'true'
    
    if args.local_only:
        os.environ['LOCAL_ONLY'] = 'true'

    logger.info("=" * 80)
    logger.info("        WILO CLOUD MONITORING & LOCAL STORAGE - COMPREHENSIVE BULK GENERATOR")
    logger.info("=" * 80)
    logger.info(f"👉 Target Normal Intervals: {args.normal_intervals}")
    logger.info(f"👉 Target Fault Runs:       {args.fault_runs} per generator (Total runs: {args.fault_runs * len(FAULT_GENERATORS)})")
    logger.info("⚡ Execution Mode:           Fast (0s generation sleep)")
    if args.local_only:
        logger.info("💾 Data Storage:             LOCAL-ONLY (Local JSONL files in ./data, cloud bypassed)")
    else:
        logger.info("💾 Data Storage:             Dual mode (Neon PostgreSQL Cloud + Local JSONL files in ./data)")
    logger.info("=" * 80)

    # Verify database connection (skip if local-only)
    if not args.local_only:
        try:
            from database import get_connection
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            conn.close()
            logger.info("✅ Database connection verified successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Database connection failed: {e}")
            logger.warning("Generation will run in LOCAL-ONLY mode (falling back gracefully).")
            os.environ['LOCAL_ONLY'] = 'true'

    start_time = time.time()
    
    # Part 1: Generate Normal Operations Data
    normal_count = args.normal_intervals
    if normal_count > 0:
        logger.info(f"\n🟢 [1/2] Generating Healthy Normal Operations Data ({normal_count} intervals)...")
        healthy_gen = HealthyGenerator()
        
        for idx in range(1, normal_count + 1):
            try:
                # Randomize load factor to cover diverse operating points
                import numpy as np
                healthy_gen.load_factor = float(np.random.uniform(0.6, 1.1))
                
                healthy_gen.generate_interval()
                
                if idx % 100 == 0 or idx == normal_count:
                    pct = (idx / normal_count) * 100
                    print(f"   [Normal Operations] Generated {idx}/{normal_count} intervals ({pct:.1f}%)")
                    sys.stdout.flush()
            except Exception as e:
                logger.error(f"Error generating normal interval {idx}: {e}")
        logger.info("✅ Normal Operations generation completed.")

    # Part 2: Generate Fault Event Data
    fault_runs = args.fault_runs
    if fault_runs > 0:
        total_fault_runs = len(FAULT_GENERATORS) * fault_runs
        logger.info(f"\n🔴 [2/2] Generating Fault Event sequences ({total_fault_runs} total runs)...")
        run_counter = 0
        
        for fault_name, generator_class in FAULT_GENERATORS:
            logger.info(f"👉 Starting sequence: {fault_name}")
            
            for iter_idx in range(1, fault_runs + 1):
                run_counter += 1
                iter_start = time.time()
                
                try:
                    # Instantiating sets the load factor randomly (0.6x to 1.1x)
                    generator = generator_class()
                    generator.run_indefinitely()
                    
                    elapsed = time.time() - iter_start
                    pct = (run_counter / total_fault_runs) * 100
                    print(f"   [{run_counter:03d}/{total_fault_runs:03d}] Run {iter_idx:02d}/{fault_runs:02d} completed | "
                          f"duration={elapsed:.2f}s | "
                          f"load={generator.load_factor:.2f}x | "
                          f"Progress: {pct:.1f}%")
                    sys.stdout.flush()
                except Exception as e:
                    logger.error(f"Error running fault generator '{fault_name}' at iteration {iter_idx}: {e}")

    # Cleanup connection pool wrapper if active
    try:
        import database
        logger.info("Flushing local storage buffer...")
        database.flush_local_storage()
        
        if getattr(database, '_GLOBAL_CONN', None) is not None:
            if hasattr(database._GLOBAL_CONN, 'real_close'):
                database._GLOBAL_CONN.real_close()
            else:
                database._GLOBAL_CONN.close()
            logger.info("✓ Closed shared database connection.")
            
        if getattr(database, '_GLOBAL_NORMAL_OPS_CONN', None) is not None:
            if hasattr(database._GLOBAL_NORMAL_OPS_CONN, 'real_close'):
                database._GLOBAL_NORMAL_OPS_CONN.real_close()
            else:
                database._GLOBAL_NORMAL_OPS_CONN.close()
            logger.info("✓ Closed shared normal operations database connection.")
    except Exception as e:
        logger.warning(f"Connection cleanup warning: {e}")

    total_elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("🎉 BULK TRAINING DATA GENERATION COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"   Total Time Elapsed: {total_elapsed/60:.2f} minutes")
    logger.info(f"   Local Mirror Folder: ./data")
    logger.info(f"   Cloud Database:      Synchronized (Neon PostgreSQL)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
