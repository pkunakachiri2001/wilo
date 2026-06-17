"""
Healthy Generator — Models the 100% normal operating baseline.
Used to generate clean data for training unsupervised anomaly detection models (e.g. Isolation Forest).
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
    SHAFT_FREQ,
)


class HealthyGenerator(BaseGenerator):
    """Models healthy normal operating baseline (no faults, always healthy)."""

    def __init__(self):
        super().__init__('Healthy Operation')
        self.is_healthy = True
        self.logger.info("Healthy operations generator initialized.")

    def generate_acceleration_data(self):
        """Steady rotational vibration at 1x shaft frequency with normal background noise."""
        t = self.generate_time_array()
        N = len(t)
        
        one_x = NOM_ACCEL_MS2 * self.load_factor * np.sin(2 * np.pi * SHAFT_FREQ * t)
        noise = np.random.normal(0, NOM_ACCEL_MS2 * self.load_factor * 0.04, N)
        return one_x + noise

    def generate_current_data(self):
        """Steady motor current matching rated operating load with minor fluctuations."""
        t = self.generate_time_array()
        N = len(t)
        fla = NOM_CURRENT_A * self.load_factor
        
        noise = np.random.normal(0, fla * 0.03, N)
        return np.clip(fla + noise, 1, 100)

    def generate_audio_data(self):
        """Steady rotational acoustics at shaft frequency plus background room noise."""
        t = self.generate_time_array()
        N = len(t)
        base = NOM_AUDIO_DB * self.load_factor ** 0.05
        
        shaft_hum = 4.0 * np.sin(2 * np.pi * SHAFT_FREQ * t)
        noise = np.random.normal(0, 1.5, N)
        return base + shaft_hum + noise

    def generate_interval(self):
        """Generate one interval and write stats. Always returns True to allow continuous running."""
        timestamps = self.generate_timestamps()
        self.interval_count += 1
        
        accel_data = self.generate_acceleration_data()
        current_data = self.generate_current_data()
        audio_data = self.generate_audio_data()
        
        # Save interval stats (writes locally to stats.json)
        self._save_interval_stats(accel_data, current_data, audio_data)
        
        # Save statistics and raw datapoints to Neon database
        self._save_interval_to_db(accel_data, current_data, audio_data)
        
        self.logger.info(
            f"Interval {self.interval_count}: Generated healthy sensor data | "
            f"Load: {self.load_factor:.2f}x"
        )
        return True
