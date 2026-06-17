#!/usr/bin/env python3
"""
High-Speed Healthy Data Generator
Generates clean normal operating baseline statistics for training Isolation Forest models.
Usage: python run_healthy_generation.py --count <number_of_intervals>
Example: python run_healthy_generation.py --count 500
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup import path
SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(SERVER_DIR.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HealthyDataPipeline")

# Import healthy generator
from fault_generators.healthy_generator import HealthyGenerator
import fault_generators.base_generator as bg


def main():
    parser = argparse.ArgumentParser(description="Generate normal healthy operational baseline data.")
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of intervals (data points) to generate (default: 500)"
    )
    args = parser.parse_args()

    count = args.count
    if count <= 0:
        logger.error("Count must be a positive integer.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🚀 Starting High-Speed Normal Operations Data Pipeline")
    logger.info(f"👉 Target: Generate {count} intervals")
    logger.info("=" * 60)

    # Disable sleep interval between generations to run at maximum speed
    bg.GENERATION_INTERVAL = 0.0
    logger.info("⚡ Set generation interval sleep to 0.0s for high throughput.")

    # Instantiate generator
    generator = HealthyGenerator()
    
    # Save the database connection wrapper details
    try:
        from database import get_connection
        conn = get_connection()
        # Verify connection
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        conn.close()
        logger.info("✅ Database connection verified.")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error("Ensure DATABASE_URL is set in your .env file.")
        sys.exit(1)

    success_count = 0
    try:
        for i in range(1, count + 1):
            # Roll a new load factor for every interval to populate the ML model with load diversity (0.6x to 1.1x)
            import numpy as np
            generator.load_factor = float(np.random.uniform(0.6, 1.1))
            
            # Run one interval
            should_continue = generator.generate_interval()
            success_count += 1
            
            # Print periodic progress
            if i % 10 == 0 or i == count:
                pct = (i / count) * 100
                logger.info(f"📊 Progress: {i}/{count} ({pct:.1f}%) intervals generated successfully.")
                
            if not should_continue:
                logger.warning("Generator signaled to stop.")
                break
                
    except KeyboardInterrupt:
        logger.info("\n⚠️ Generation interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Critical error during generation: {e}", exc_info=True)
    finally:
        try:
            import database
            logger.info("Flushing local storage buffer...")
            database.flush_local_storage()
            
            if getattr(database, '_GLOBAL_NORMAL_OPS_CONN', None) is not None:
                if hasattr(database._GLOBAL_NORMAL_OPS_CONN, 'real_close'):
                    database._GLOBAL_NORMAL_OPS_CONN.real_close()
                else:
                    database._GLOBAL_NORMAL_OPS_CONN.close()
                logger.info("✓ Closed shared normal operations database connection.")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
            
        logger.info("=" * 60)
        logger.info("🏁 Normal Operations Data Generation Completed!")
        logger.info(f"   Generated intervals: {success_count} / {count}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
