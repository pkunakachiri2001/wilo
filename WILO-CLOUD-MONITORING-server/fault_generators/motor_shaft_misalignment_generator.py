"""
Motor Shaft Misalignment Generator — GRADUAL FAULT

Physics:
  Angular misalignment: 1× RPM dominant; high axial vibration; phase difference
    between driver and driven shafts.
  Parallel misalignment: 2× RPM dominant; radial vibration in direction of offset.

  For a mixed/typical case (both angular + parallel components):
    • 1× component grows in amplitude with severity (angular contribution)
    • 2× component grows faster (parallel contribution dominates in radial direction)
    • High kurtosis is NOT characteristic — misalignment is a deterministic sine,
      so kurtosis stays near 1.5 (pure sine wave)
    • Current rises modestly from increased bearing load

  Frequency model (730 RPM motor):
    1× = SHAFT_FREQ ≈ 12.17 Hz
    2× = 2 × SHAFT_FREQ ≈ 24.3 Hz

  Phase randomised per run — real misalignment has a fixed but unknown phase
  relative to the measurement trigger.
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    SHAFT_FREQ, TWO_X_LINE,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
)


class MotorShaftMisalignmentGenerator(BaseGenerator):
    """Motor Shaft Misalignment — progressive angular + parallel offset."""

    def __init__(self):
        super().__init__('Motor Shaft Misalignment')
        self.critical_interval = np.random.randint(11, 16)

        # Random phase angles per run (real misalignment has fixed but unknown phase)
        self._phi1 = np.random.uniform(0, 2 * np.pi)   # 1× phase
        self._phi2 = np.random.uniform(0, 2 * np.pi)   # 2× phase

        # Split between angular vs parallel character (randomised per run)
        self._angular_frac = np.random.uniform(0.3, 0.7)   # fraction of 1× vs 2×

        self.logger.info(
            f"Misalignment: critical_interval={self.critical_interval}  "
            f"angular_frac={self._angular_frac:.2f}  "
            f"phi1={np.degrees(self._phi1):.0f}°  phi2={np.degrees(self._phi2):.0f}°"
        )

    def _severity(self) -> float:
        return float(np.clip(self.interval_count / self.critical_interval, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        """
        Both 1× and 2× components grow proportionally to misalignment severity.
        For angular misalignment the 1× dominates; for parallel the 2× dominates.
        Background Gaussian noise is relatively small (misalignment is deterministic).
        Kurtosis stays near 1.5 (pure sinusoidal — NOT impulsive).
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()

        base = NOM_ACCEL_MS2 * self.load_factor

        if self.system_failure_state:
            # Critical misalignment: chaotic multi-harmonic vibration as coupling fails
            one_x   = (2.5 * self._angular_frac)       * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi1)
            two_x   = (2.5 * (1 - self._angular_frac)) * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t + self._phi2)
            three_x = 0.8  * np.sin(2 * np.pi * 3 * SHAFT_FREQ * t)
            noise   = np.random.normal(0, 0.6, N)
            return base + one_x + two_x + three_x + noise

        # 1× amplitude grows with severity (angular component)
        A1 = self._angular_frac * (0.2 + 2.8 * sev) * self.load_factor
        # 2× amplitude grows slightly faster (parallel component)
        A2 = (1 - self._angular_frac) * (0.15 + 3.2 * sev) * self.load_factor

        one_x = A1 * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi1)
        two_x = A2 * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t + self._phi2)
        noise = np.random.normal(0, base * 0.06 + 0.08 * sev, N)

        data = base + one_x + two_x + noise

        if self.interval_count >= self.critical_interval and not self.system_failure_state:
            self.system_failure_state = True
            self.logger.warning(f"🔄 MISALIGNMENT FAILURE at interval {self.interval_count}!")

        return data

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        """
        Misalignment increases bearing side-load → moderate current rise.
        Also creates a periodic torque variation at 2× shaft frequency,
        which appears as small current modulation at 2× RPM.
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        fla = NOM_CURRENT_A * self.load_factor

        if self.system_failure_state:
            # Coupling failure: heavy load oscillation
            base = fla * 3.2
            modulation = fla * 0.8 * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t)
            noise = np.random.normal(0, 2.5, N)
            return np.clip(base + modulation + noise, fla, fla * 6)

        # Torque modulation at 2× shaft frequency (proportional to misalignment)
        modulation = fla * 0.15 * sev * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t + self._phi2)
        dc_rise    = fla * (1.0 + 0.25 * sev)   # friction load increases ~25% at full severity
        noise      = np.random.normal(0, fla * 0.025, N)
        return np.clip(dc_rise + modulation + noise, 1, 100)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        """
        Misalignment audio: deterministic tonal component at 2× shaft frequency
        (rumbling/thumping) that grows with severity. Low kurtosis — tonal, not impulsive.
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()

        base = NOM_AUDIO_DB * self.load_factor ** 0.05

        if self.system_failure_state:
            # Coupling failure: loud low-frequency thumping
            low_thump  = 14.0 * np.sin(2 * np.pi * SHAFT_FREQ * t)
            two_x_tone = 10.0 * np.sin(2 * np.pi * 2 * SHAFT_FREQ * t)
            noise      = np.random.normal(0, 3.0, N)
            return base + 8 + low_thump + two_x_tone + noise

        one_x_tone = (1.5 + 10.0 * sev * self._angular_frac) * np.sin(
            2 * np.pi * SHAFT_FREQ * t + self._phi1)
        two_x_tone = (1.0 + 12.0 * sev * (1 - self._angular_frac)) * np.sin(
            2 * np.pi * 2 * SHAFT_FREQ * t + self._phi2)
        noise = np.random.normal(0, 1.0 + 1.5 * sev, N)

        return base + one_x_tone + two_x_tone + noise


if __name__ == '__main__':
    generator = MotorShaftMisalignmentGenerator()
    generator.run_indefinitely()
