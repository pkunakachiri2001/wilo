"""
Motor Vibration Anomaly Generator — GRADUAL FAULT (with 3 physically distinct sub-types)

Physics:
  Vibration anomalies have THREE physically distinct origins with different signatures.
  Each run randomly selects one sub-type to create ML training diversity:

  Sub-type A — IMBALANCE (mass eccentricity):
    • Large 1× RPM amplitude, constant phase, no harmonics
    • Very low kurtosis (~1.5, pure sine wave)
    • Current: small 1× modulation from torque ripple
    • Diagnosis: replace/rebalance rotor

  Sub-type B — STRUCTURAL LOOSENESS (worn bearing housing, loose bolts):
    • Sub-harmonics: 0.5×, 1×, 1.5×, 2× RPM present simultaneously
    • Phase is unstable (wobbles ±20°)
    • Moderate kurtosis (5–10) — not highly impulsive
    • Truncation in waveform (clipping against loose surfaces)
    • Audio: knocking at 0.5× shaft frequency

  Sub-type C — DRY FRICTION / RUBS:
    • Broadband excitation with peaks at structural natural frequencies
    • Very high kurtosis (>15) — highly impulsive
    • Random burst character — NOT periodic
    • Audio: harsh, broadband, high-energy noise bursts
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    impact_train, damped_impact,
    SHAFT_FREQ, TWO_X_LINE,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
    STRUCT_OMEGA_N, STRUCT_ZETA,
)

_SUB_TYPES = ['imbalance', 'looseness', 'friction']


class MotorVibrationAnomalyGenerator(BaseGenerator):
    """Motor Vibration Anomaly — 3 physically distinct sub-types chosen per run."""

    def __init__(self):
        super().__init__('Motor Vibration Anomaly')
        self.critical_interval = np.random.randint(11, 16)

        # Select sub-type for this run
        self._sub_type = np.random.choice(_SUB_TYPES)

        # Random phase for imbalance / looseness (fixed per run)
        self._phi = np.random.uniform(0, 2 * np.pi)

        # Looseness: unstable phase jitter amplitude
        self._phase_jitter_sigma = np.random.uniform(0.05, 0.25)

        self.logger.info(
            f"Vibration anomaly sub-type: {self._sub_type.upper()}  "
            f"critical_interval={self.critical_interval}  "
            f"load_factor={self.load_factor:.2f}"
        )

    def _severity(self) -> float:
        return float(np.clip(self.interval_count / self.critical_interval, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        base = NOM_ACCEL_MS2 * self.load_factor

        if self.system_failure_state:
            # All sub-types reach chaotic high-energy failure
            signal  = base * 2
            signal += 3.0 * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi)
            signal += 2.0 * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t)
            signal += impact_train(t, SHAFT_FREQ * 0.5, 2.5, amplitude_sigma=0.5,
                                   omega_n=STRUCT_OMEGA_N, zeta=STRUCT_ZETA)
            signal += np.random.normal(0, 1.0, N)
            return signal

        data = np.full(N, base)
        noise = np.random.normal(0, base * 0.05, N)

        if self._sub_type == 'imbalance':
            # Pure 1× sine with growing amplitude; kurtosis stays ~1.5
            A1 = base * (0.2 + 4.0 * sev)
            data += A1 * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi)
            data += noise

        elif self._sub_type == 'looseness':
            # Sub-harmonics: 0.5×, 1×, 1.5×, 2×, 3× with unstable phase
            phase_jitter = self._phase_jitter_sigma * np.random.randn()
            A_base = base * 0.6 * sev
            data += A_base * 1.5 * np.sin(2 * np.pi * 0.5 * SHAFT_FREQ * t + self._phi)
            data += A_base * 2.0 * np.sin(2 * np.pi *       SHAFT_FREQ * t + self._phi + phase_jitter)
            data += A_base * 1.2 * np.sin(2 * np.pi * 1.5 * SHAFT_FREQ * t)
            data += A_base * 0.8 * np.sin(2 * np.pi * 2.0 * SHAFT_FREQ * t)
            data += A_base * 0.4 * np.sin(2 * np.pi * 3.0 * SHAFT_FREQ * t)
            # Clipping: looseness impacts a hard stop → asymmetric waveform
            data = np.clip(data, -999, base + base * (0.5 + 2.0 * sev))
            data += noise * 2

        elif self._sub_type == 'friction':
            # High-kurtosis random bursts at structural natural frequencies
            burst_rate = 0.003 + 0.015 * sev
            burst_events = np.random.binomial(1, burst_rate, N).astype(bool)
            friction_signal = np.zeros(N, dtype=float)
            for idx in np.where(burst_events)[0]:
                A_burst = np.random.lognormal(np.log(0.5 + 2.5 * sev), 0.45)
                friction_signal += damped_impact(t, t[idx], A_burst,
                                                  omega_n=STRUCT_OMEGA_N,
                                                  zeta=STRUCT_ZETA)
            data += friction_signal + noise

        if self.interval_count >= self.critical_interval and not self.system_failure_state:
            self.system_failure_state = True
            self.logger.warning(f"📈 VIBRATION ANOMALY ({self._sub_type}) "
                                f"FAILURE at interval {self.interval_count}!")

        return data

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        fla = NOM_CURRENT_A * self.load_factor

        if self.system_failure_state:
            modulation = fla * 0.6 * np.sin(2 * np.pi * SHAFT_FREQ * t)
            noise = np.random.normal(0, 4.0, N)
            return np.clip(fla * 2.8 + modulation + noise, fla, fla * 6)

        if self._sub_type == 'imbalance':
            # Torque ripple at 1× → small current modulation
            mod = fla * 0.08 * sev * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi)
            noise = np.random.normal(0, fla * 0.025, N)
            return np.clip(fla + mod + noise, 1, 100)

        elif self._sub_type == 'looseness':
            # Multiple sub-harmonic torque components
            mod = (fla * 0.1 * sev * np.sin(2 * np.pi * 0.5 * SHAFT_FREQ * t) +
                   fla * 0.06 * sev * np.sin(2 * np.pi * 2.0 * SHAFT_FREQ * t))
            dc_rise = fla * (1.0 + 0.2 * sev)
            noise = np.random.normal(0, fla * 0.03, N)
            return np.clip(dc_rise + mod + noise, 1, 100)

        else:  # friction
            # Random load spikes from intermittent rubbing
            friction_load = np.zeros(N, dtype=float)
            rub_events = np.random.binomial(1, 0.005 + 0.015 * sev, N).astype(bool)
            friction_load[rub_events] = np.random.lognormal(np.log(fla * 0.4 * sev), 0.4,
                                                              rub_events.sum())
            noise = np.random.normal(0, fla * 0.03, N)
            return np.clip(fla + friction_load + noise, 1, 100)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        base = NOM_AUDIO_DB * self.load_factor ** 0.05

        if self.system_failure_state:
            broad_noise = np.random.normal(0, 9.0, N)
            tone = 14.0 * np.sin(2 * np.pi * SHAFT_FREQ * t)
            return base + 10 + tone + broad_noise

        if self._sub_type == 'imbalance':
            # Tonal hum at 1× shaft — steady, no crackling
            tone = (2.0 + 9.0 * sev) * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi)
            noise = np.random.normal(0, 1.5, N)
            return base + tone + noise

        elif self._sub_type == 'looseness':
            # Knocking at 0.5× shaft + broadband noise
            knock = (3.0 + 8.0 * sev) * np.sin(2 * np.pi * 0.5 * SHAFT_FREQ * t)
            noise = np.random.normal(0, 2.0 + 3.0 * sev, N)
            return base + knock + noise

        else:  # friction
            # Harsh broadband noise bursts (high kurtosis audio)
            rub_events = np.random.binomial(1, 0.004 + 0.014 * sev, N).astype(bool)
            rub_noise = np.random.exponential(2.0 + 6.0 * sev, N) * rub_events
            background = np.random.normal(0, 2.0 + 3.0 * sev, N)
            return base + rub_noise + background


if __name__ == '__main__':
    generator = MotorVibrationAnomalyGenerator()
    generator.run_indefinitely()
