"""
Motor Bearing Failure Generator — GRADUAL FAULT (ISO 13373-3 four-stage model)

Physics:
  Bearing degradation follows four distinct stages:
    Stage 1 (intervals 1–3):   Incipient — no detectable change; kurtosis ≈ 3
    Stage 2 (intervals 4–8):   Minor wear — isolated BPFO/BPFI impacts; kurtosis rises 3→15
    Stage 3 (intervals 9–12):  Moderate — sideband bursts; kurtosis peaks 15→28 then drops
    Stage 4 (intervals 13–15): Critical  — widespread damage; kurtosis 6–8, RMS × 5–6×

Signal structure:
  accel = background_noise
        + impact_train(BPFO, A_bpfo)   ← outer race defect frequency
        + impact_train(BPFI, A_bpfi)   ← inner race defect frequency (modulated at 1× shaft)
        + 1× shaft component (residual imbalance)

  current = nominal_current × load_factor  (bearing friction increases slip → slight current rise)
  audio   = background_hum + bearing_squeal_tone + impact_noise
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    impact_train, damped_impact,
    SHAFT_FREQ, BPFO, BPFI, BSF,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
    STRUCT_OMEGA_N, STRUCT_ZETA,
    TWO_X_LINE,
)


class MotorBearingFailureGenerator(BaseGenerator):
    """Motor Bearing Failure — four-stage ISO 13373-3 degradation model."""

    def __init__(self):
        super().__init__('Motor Bearing Failure')

        # Randomise the critical interval slightly per run for ML variety
        self.critical_interval = np.random.randint(12, 16)

        # Stage boundaries (fraction of critical_interval)
        self._s1_end = 0.25   # Stage 1: 0–25%
        self._s2_end = 0.55   # Stage 2: 25–55%
        self._s3_end = 0.82   # Stage 3: 55–82%
        # Stage 4: 82–100% → failure

        # Outer-race defect impacts are always present once Stage 2 begins;
        # inner-race impacts grow later and are modulated by shaft speed.
        # Store amplitude scale so load_factor drives a consistent baseline.
        self._base_accel = NOM_ACCEL_MS2 * self.load_factor

        self.logger.info(
            f"Bearing: 4-stage model  critical_interval={self.critical_interval}  "
            f"BPFO={BPFO:.1f} Hz  BPFI={BPFI:.1f} Hz  "
            f"load_factor={self.load_factor:.2f}"
        )

    # ------------------------------------------------------------------
    # Stage helper
    # ------------------------------------------------------------------
    def _stage(self) -> int:
        """Return current degradation stage (1–4)."""
        frac = self.interval_count / self.critical_interval
        if frac <= self._s1_end:
            return 1
        elif frac <= self._s2_end:
            return 2
        elif frac <= self._s3_end:
            return 3
        return 4

    def _severity(self) -> float:
        """Normalised progression 0→1 through Stage 2+3 (used for amplitude scaling)."""
        frac = self.interval_count / self.critical_interval
        return float(np.clip((frac - self._s1_end) / (1.0 - self._s1_end), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        """
        4-stage vibration with physically correct defect-frequency impacts.

        Kurtosis profile (approximate, driven by impact amplitude):
          Stage 1: ~3   (Gaussian noise only)
          Stage 2: 3→15 (isolated BPFO transients)
          Stage 3: peak ~25 then drops to ~8 (damage spreads — kurtosis falls)
          Stage 4: 5–8, but RMS×5 (widespread spall, high energy)
        """
        t = self.generate_time_array()
        stage = self._stage()
        sev   = self._severity()

        # -- Background: baseline vibration at 1× shaft (residual imbalance) --------
        baseline_amp = self._base_accel * (1.0 + 0.15 * sev)
        one_x = 0.15 * self._base_accel * np.sin(2 * np.pi * SHAFT_FREQ * t)
        noise = np.random.normal(0, 0.04 * self._base_accel + 0.02 * sev, len(t))

        signal = baseline_amp + one_x + noise

        if self.system_failure_state:
            # Stage 4 collapse: widespread spall → broadband high-energy noise
            # RMS grows ×5–6, kurtosis falls back toward 5
            spall_rms = 5.5 * self._base_accel
            # High-amplitude broad impact train at BPFO (many overlapping impacts)
            signal += impact_train(t, BPFO, spall_rms * 0.6, amplitude_sigma=0.6,
                                   omega_n=STRUCT_OMEGA_N, zeta=0.12)
            signal += impact_train(t, BPFI, spall_rms * 0.5, amplitude_sigma=0.6,
                                   omega_n=STRUCT_OMEGA_N * 1.3, zeta=0.12)
            signal += np.random.normal(0, spall_rms * 0.35, len(t))
            return signal

        if stage == 1:
            # Healthy — Gaussian background only; no defect impacts
            pass

        elif stage == 2:
            # Isolated BPFO impacts growing in amplitude — kurtosis rising
            # Amplitude grows from 0 → 1.5× baseline over Stage 2
            s2_frac = np.clip(
                (self.interval_count / self.critical_interval - self._s1_end)
                / (self._s2_end - self._s1_end), 0.0, 1.0
            )
            A_bpfo = 0.3 + 1.5 * s2_frac   # m/s²
            A_bpfi = 0.15 * s2_frac         # inner race lighter at this stage

            signal += impact_train(t, BPFO, A_bpfo * self.load_factor,
                                   amplitude_sigma=0.3, jitter_frac=0.04)
            # BPFI modulated by 1× shaft (amplitude wobbles at shaft rate)
            shaft_modulation = 0.5 + 0.5 * np.abs(np.sin(np.pi * SHAFT_FREQ * t))
            bpfi_raw = impact_train(t, BPFI, A_bpfi * self.load_factor,
                                    amplitude_sigma=0.35, jitter_frac=0.06)
            signal += bpfi_raw * shaft_modulation

        elif stage == 3:
            # Sideband bursts: both BPFO and BPFI prominent
            # Amplitude peaks then noise floor rises (kurtosis peaks then drops)
            s3_frac = np.clip(
                (self.interval_count / self.critical_interval - self._s2_end)
                / (self._s3_end - self._s2_end), 0.0, 1.0
            )
            # Peak amplitude at start of Stage 3, then rising noise masks impacts
            A_bpfo = (2.5 - 0.8 * s3_frac) * self.load_factor
            A_bpfi = (2.0 - 0.6 * s3_frac) * self.load_factor
            noise_floor = 0.3 + 1.2 * s3_frac

            signal += impact_train(t, BPFO, A_bpfo, amplitude_sigma=0.4)
            signal += impact_train(t, BPFI, A_bpfi, amplitude_sigma=0.45)
            signal += np.random.normal(0, noise_floor, len(t))

        # Trigger failure at/beyond critical interval
        if self.interval_count >= self.critical_interval and not self.system_failure_state:
            self.system_failure_state = True
            self.logger.warning(f"🔥 BEARING FAILURE at interval {self.interval_count}!")

        return signal

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        """
        Bearing friction increases mechanical load → slight slip increase
        → small, steady current rise throughout degradation.
        Current signature is dominated by noise, NOT a large DC step.
        """
        t = self.generate_time_array()
        sev = self._severity()

        if self.system_failure_state:
            # Seized bearing: motor tries to restart against locked load
            base = NOM_CURRENT_A * self.load_factor * 4.5
            noise = np.random.normal(0, 2.0, len(t))
            return np.clip(base + noise, 40, 120)

        # Gradual: 5% rise over full degradation (friction increases slip)
        base = NOM_CURRENT_A * self.load_factor * (1.0 + 0.05 * sev)
        noise = np.random.normal(0, base * 0.02, len(t))
        return np.clip(base + noise, 5, 100)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        """
        Stage-dependent bearing sound:
          Stage 1: quiet background hum
          Stage 2: faint periodic clicking at BPFO
          Stage 3: audible grinding, tonal BPFO component grows
          Stage 4: loud broadband grinding + continuous squeal
        """
        t = self.generate_time_array()
        stage = self._stage()
        sev   = self._severity()

        base = NOM_AUDIO_DB * self.load_factor ** 0.1  # dB barely changes with load
        noise = np.random.normal(0, 1.0 + 2.0 * sev, len(t))
        signal = base + noise

        if self.system_failure_state:
            squeal = 18.0 * np.sin(2 * np.pi * 90 * t)  # bearing squeal ~90 Hz
            grinding = np.random.normal(0, 6.0, len(t))
            return base + 12 + squeal + grinding

        if stage >= 2:
            # Periodic clicking at BPFO (gentle audio impact)
            click_amp = 1.5 * sev
            signal += impact_train(t, BPFO, click_amp,
                                   omega_n=2 * np.pi * 300, zeta=0.15,
                                   amplitude_sigma=0.3)

        if stage >= 3:
            # Audible grind: 2× BPFO tone + broadband grind
            signal += 4.0 * sev * np.sin(2 * np.pi * 2 * BPFO * t)
            signal += np.random.normal(0, 2.5 * sev, len(t))

        return signal


if __name__ == '__main__':
    generator = MotorBearingFailureGenerator()
    generator.run_indefinitely()
