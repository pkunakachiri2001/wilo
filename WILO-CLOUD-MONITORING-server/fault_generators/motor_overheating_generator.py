"""
Motor Overheating Generator — GRADUAL FAULT (thermal degradation)

Physics:
  Temperature follows Newton's law of cooling (first-order exponential approach):
    T(t) = T_ambient + ΔT_ss × (1 − exp(−t / τ_thermal))
  where τ_thermal is the motor thermal time constant (compressed to intervals here).

  Copper winding resistance: R(T) = R₀ × (1 + α_Cu × ΔT)  [α_Cu = 0.00393 /°C]

  Current behaviour at constant mechanical load (induction motor):
    As T rises → R_winding increases → slip increases to maintain torque
    → small current rise (≤10% for moderate overheating).
    The dominant electrical signatures are INCREASED HARMONICS and NOISE,
    not a large DC step (that would only occur for severe overheating/insulation failure).

  Audio:
    Cooling fan noise increases (fan is thermal-regulated).
    Thermal stress = low-frequency thermal cycling hum.
    At critical: winding insulation breakdown → electrical buzz at 2× line (100 Hz).
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    SHAFT_FREQ, TWO_X_LINE, THREE_X_LINE,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
)


class MotorOverheatingGenerator(BaseGenerator):
    """Motor Overheating — exponential thermal degradation model."""

    def __init__(self):
        super().__init__('Motor Overheating')
        self.critical_interval = np.random.randint(11, 16)

        # Motor thermal time constant (compressed from hours to intervals).
        # The winding temperature rises quickly at first, then plateaus.
        self._tau = self.critical_interval * 0.4   # ~40% of total run

        # Temperature rise above ambient at thermal saturation (°C)
        self._delta_T_ss = np.random.uniform(80, 130)   # e.g. 105°C above ambient

        self.logger.info(
            f"Overheating: critical_interval={self.critical_interval}  "
            f"ΔT_ss={self._delta_T_ss:.0f}°C  tau={self._tau:.1f} intervals"
        )

    def _temp_rise(self) -> float:
        """Current temperature rise above ambient (°C)."""
        return self._delta_T_ss * (1 - np.exp(-self.interval_count / self._tau))

    def _severity(self) -> float:
        """Normalised thermal severity 0→1."""
        return float(np.clip(self._temp_rise() / self._delta_T_ss, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        """
        Thermal expansion increases rotor-stator clearance → small, steady
        vibration rise at 1× shaft. NOT impulsive — purely deterministic.
        Only at critical temperature does insulation arcing add impulse spikes.
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        base = NOM_ACCEL_MS2 * self.load_factor

        if self.system_failure_state:
            # Winding insulation breakdown: electromagnetic excitation transients
            em_vibration = 0.6 * np.sin(2 * np.pi * TWO_X_LINE * t)
            thermal_sway = 1.8 * np.sin(2 * np.pi * 2.5 * t)   # very slow thermal wobble
            noise = np.random.normal(0, 0.7, N)
            return base * 4 + em_vibration + thermal_sway + noise

        # Thermal expansion → slight 1× amplitude growth + slow thermal ripple
        one_x = (base * 0.3 + base * 0.8 * sev) * np.sin(2 * np.pi * SHAFT_FREQ * t)
        thermal_ripple = 0.5 * sev * np.sin(2 * np.pi * 1.5 * t)   # ~1.5 Hz thermal cycling
        noise = np.random.normal(0, base * 0.05 + 0.1 * sev, N)
        data = base + one_x + thermal_ripple + noise

        if self.interval_count >= self.critical_interval and not self.system_failure_state:
            self.system_failure_state = True
            self.logger.warning(f"🔥 OVERHEATING FAILURE at interval {self.interval_count}! "
                                f"T_rise={self._temp_rise():.0f}°C")

        return data

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        """
        Correct thermal current model:
          At constant load, R_winding rises → slip increases → current rises ~5–10%.
          Dominant electrical signature is HARMONICS (especially 3× line = 150 Hz),
          not a large DC step. DC step only occurs during insulation failure.
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        fla = NOM_CURRENT_A * self.load_factor

        if self.system_failure_state:
            # Insulation breakdown: phase imbalance → strong harmonics + current surge
            base = fla * 2.8
            harmonic_3x = fla * 0.4 * np.sin(2 * np.pi * THREE_X_LINE * t)
            noise = np.random.normal(0, 3.5, N)
            return np.clip(base + harmonic_3x + noise, fla, fla * 7)

        # Thermal slip increase: small DC rise (5% max) + growing harmonic noise
        dc_rise = fla * (1.0 + 0.05 * sev)
        # Thermal harmonic: 3× line grows with temperature (insulation stress)
        h3 = fla * 0.06 * sev * np.sin(2 * np.pi * THREE_X_LINE * t)
        # Slow thermal cycling appears as very-low-frequency modulation
        thermal_cycling = fla * 0.03 * sev * np.sin(2 * np.pi * 0.8 * t)
        noise = np.random.normal(0, fla * 0.02, N)
        return np.clip(dc_rise + h3 + thermal_cycling + noise, 1, 100)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        """
        Cooling fan ramps up with temperature (thermostat-controlled).
        Critical: winding arcing creates 100 Hz electrical buzz.
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        temp_C = self._temp_rise()

        base = NOM_AUDIO_DB * self.load_factor ** 0.05

        if self.system_failure_state:
            # Winding arc + cooling fan at max
            arc_buzz = 20.0 * np.sin(2 * np.pi * TWO_X_LINE * t)
            fan_hiss = np.random.normal(0, 5.0, N)
            return base + 12 + arc_buzz + fan_hiss

        # Fan noise grows with temperature (louder cooling at higher temp)
        fan_level = 5.0 * sev   # dB increase from fan ramping
        fan_hiss = np.random.normal(0, 1.5 + 3.0 * sev, N)

        # Thermal hum at 2× line (100 Hz) grows with winding temperature
        thermal_hum = 4.0 * sev * np.sin(2 * np.pi * TWO_X_LINE * t)

        noise = np.random.normal(0, 1.0, N)
        return base + fan_level + thermal_hum + fan_hiss + noise


if __name__ == '__main__':
    generator = MotorOverheatingGenerator()
    generator.run_indefinitely()
