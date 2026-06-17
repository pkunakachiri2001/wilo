"""
Pump Seal Leakage Generator — SUDDEN FAULT (mechanical seal failure)

Physics (DIFFERENTIATED from cavitation — distinct fluid-dynamics model):

  Mechanical seal failure creates three simultaneous effects:

  1. EXTERNAL LEAK PATH (fluid escapes around shaft):
     → Broadband high-frequency HISSING audio (300–3000 Hz turbulent flow)
     → The hiss is the primary distinguishing signature from cavitation

  2. SUCTION-SIDE PRESSURE DROP (fluid loss reduces net positive suction head):
     → Onset of secondary cavitation at the impeller inlet
     → Lower-intensity BPF-synchronised impacts (secondary, not primary signature)

  3. MECHANICAL VIBRATION CHANGE:
     → Seal contact loss reduces mechanical damping → slight vibration increase
     → No impulsive signature — just elevated broadband noise floor

  Current behaviour:
     → As fluid escapes, pump runs "light" (less hydraulic load) → current briefly drops
     → Then, as the pump tries to maintain pressure, current becomes erratic

  This is DISTINCT from cavitation in:
     • Audio: hissing (turbulent leak) vs. popping (bubble collapse)
     • Vibration: broadband noise vs. BPF-synchronised impacts
     • Current: initial drop (unloading) vs. irregular spikes (cavitation head loss)
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    impact_train,
    BPF, SHAFT_FREQ,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
    STRUCT_OMEGA_N, STRUCT_ZETA,
)


class PumpSealLeakageGenerator(BaseGenerator):
    """Pump Seal Leakage — turbulent leak hissing + secondary cavitation model."""

    def __init__(self):
        super().__init__('Pump Seal Leakage')
        self.is_healthy = True
        self.spike_interval = np.random.randint(10, 16)
        # Randomise leak severity per run (partial vs. full seal failure)
        self._leak_fraction = np.random.uniform(0.3, 1.0)   # fraction of seal circumference failed
        self.logger.info(
            f"Seal leakage at interval {self.spike_interval}  "
            f"leak_fraction={self._leak_fraction:.2f}"
        )

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        """
        Primary: elevated broadband noise floor from reduced mechanical damping.
        Secondary: mild BPF impacts from suction cavitation (not primary signature).
        NOT a large impulse spike — seal leakage is NOT a bearing impact event.
        """
        t = self.generate_time_array()
        N = len(t)

        if self.system_failure_state:
            # Full seal failure: reduced damping + secondary cavitation noise
            base = NOM_ACCEL_MS2 * self.load_factor * 2.2
            broad_noise = np.random.normal(0, base * 0.6, N)
            # Mild suction cavitation impacts at BPF (secondary)
            suction_cav = impact_train(t, BPF, 0.8 * self._leak_fraction * self.load_factor,
                                       amplitude_sigma=0.6, omega_n=2 * np.pi * 2000, zeta=0.07)
            return base + broad_noise + suction_cav

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            self.failure_triggered = True
            self.system_failure_state = True
            self.is_healthy = False
            self.logger.warning(
                f"🔧 SEAL FAILURE at interval {self.interval_count}! "
                f"Leak fraction={self._leak_fraction:.2f}  "
                f"Suction cavitation beginning."
            )
            # Rupture moment: brief transient from seal contact loss + pressure wave
            base = NOM_ACCEL_MS2 * self.load_factor * 1.8
            # Pressure wave from seal rupture
            pressure_transient = 2.5 * self.load_factor * np.exp(-t / 0.15) * \
                                  np.sin(2 * np.pi * SHAFT_FREQ * t)
            broad_noise = np.random.normal(0, base * 0.5, N)
            return base + pressure_transient + broad_noise

        # Normal: very stable, well-damped operation
        noise = np.random.normal(0, NOM_ACCEL_MS2 * self.load_factor * 0.04, N)
        shaft_ripple = 0.1 * self.load_factor * np.sin(2 * np.pi * SHAFT_FREQ * t)
        return NOM_ACCEL_MS2 * self.load_factor + shaft_ripple + noise

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        """
        Seal leak → reduced hydraulic load → current initially DROPS,
        then becomes erratic as pump runs against reduced pressure / partial vapour lock.
        This is opposite to cavitation (which drives current UP from increased head loss).
        """
        t = self.generate_time_array()
        N = len(t)
        fla = NOM_CURRENT_A * self.load_factor

        if self.system_failure_state:
            # Pump running light then erratic (vapour ingestion oscillation)
            base = fla * 0.75   # running light due to pressure loss
            variation = fla * 0.4 * self._leak_fraction * np.sin(2 * np.pi * 0.5 * t)
            noise = np.random.normal(0, fla * 0.15, N)
            return np.clip(base + variation + noise, fla * 0.3, fla * 2.5)

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            # Brief drop: pump suddenly unloaded as fluid escapes
            drop = fla * (0.65 + 0.25 * np.exp(-t / 0.4))   # drops, then recovers
            noise = np.random.normal(0, fla * 0.1, N)
            return np.clip(drop + noise, fla * 0.3, fla * 2)

        # Normal: stable current
        noise = np.random.normal(0, fla * 0.025, N)
        return np.clip(fla + noise, 1, 100)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        """
        PRIMARY DISTINGUISHING SIGNATURE:
        Turbulent fluid escaping through the leak path creates BROADBAND HISSING
        (300–3000 Hz, continuous, non-periodic — like steam escaping a valve).
        This is completely distinct from cavitation "popcorn" pops.

        Modelled as: high-variance Gaussian noise (broadband turbulent hiss)
        amplitude-modulated by leak rate.
        """
        t = self.generate_time_array()
        N = len(t)

        if self.system_failure_state:
            base = NOM_AUDIO_DB + 8
            # Turbulent hissing: broadband Gaussian scaled by leak fraction
            turbulent_hiss = np.random.normal(0, 12.0 * self._leak_fraction, N)
            # Secondary cavitation: occasional soft pops (low amplitude)
            suction_pops = impact_train(t, BPF, 2.0 * self._leak_fraction,
                                        amplitude_sigma=0.7,
                                        omega_n=2 * np.pi * 600, zeta=0.25,
                                        jitter_frac=0.15)
            return base + turbulent_hiss + suction_pops

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            # Seal rupture sound: brief crack then immediate hissing
            base = NOM_AUDIO_DB + 12
            crack_transient = 10.0 * np.exp(-t / 0.05)   # very brief crack
            turbulent_hiss = np.random.normal(0, 10.0 * self._leak_fraction, N)
            return base + crack_transient + turbulent_hiss

        # Normal: quiet, stable pump operation
        noise = np.random.normal(0, 1.2, N)
        return NOM_AUDIO_DB * self.load_factor ** 0.04 + noise


if __name__ == '__main__':
    generator = PumpSealLeakageGenerator()
    generator.run_indefinitely()
