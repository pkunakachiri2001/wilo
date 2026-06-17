"""
Motor Winding Failure Generator — GRADUAL FAULT (insulation breakdown)

Physics:
  Turn-to-turn short circuits develop progressively in the stator winding.

  Electrical signatures (FIXED — 50 Hz system):
    • Odd harmonics in stator current: 3rd (150 Hz), 5th (250 Hz) dominant
      - Phase imbalance from shorted turns drives inter-harmonic currents
      - 3rd harmonic grows as H3 ∝ (shorted turns / total turns)²
      - 5th harmonic follows at ~60% of 3rd harmonic magnitude
    • Current waveform becomes asymmetric (skewness rises)
    • Partial discharge (PD) events: sporadic, temporally clustered Weibull bursts
      (NOT Poisson — PD is self-triggering and temporally clustered)

  Vibration signatures:
    • 2× line frequency (100 Hz) in acceleration from magnetic force imbalance
    • Growing broadband noise from electromagnetic excitation

  Audio:
    • Crackling/popping from partial discharges
    • Growing 100 Hz electromagnetic hum (NOT 120 Hz — 50 Hz system)
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    damped_impact,
    TWO_X_LINE, THREE_X_LINE, FIVE_X_LINE,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
    STRUCT_OMEGA_N, STRUCT_ZETA,
)


def _weibull_pd_events(n_samples: int, rate_per_sample: float,
                       cluster_size: float = 3.0) -> np.ndarray:
    """
    Generate partial discharge event timestamps using a Weibull-based
    clustered model (PD events are self-triggering — one discharge
    lowers the threshold for the next).

    Returns a boolean array of length n_samples: True = PD event occurred.
    """
    events = np.zeros(n_samples, dtype=bool)
    i = 0
    refractory = 0   # minimum samples between events
    while i < n_samples:
        if i < refractory:
            i += 1
            continue
        if np.random.random() < rate_per_sample:
            events[i] = True
            # After a PD event, cluster of follow-up events within short window
            n_cluster = np.random.poisson(cluster_size)
            for _ in range(n_cluster):
                j = i + np.random.randint(1, 15)
                if j < n_samples:
                    events[j] = True
            # Refractory period after cluster
            refractory = i + np.random.randint(5, 40)
        i += 1
    return events


class MotorWindingFailureGenerator(BaseGenerator):
    """Motor Winding Failure — progressive insulation breakdown with correct harmonic injection."""

    def __init__(self):
        super().__init__('Motor Winding Failure')
        self.critical_interval = np.random.randint(11, 16)

        # Random phase offsets for harmonics per run (winding geometry varies)
        self._phi3 = np.random.uniform(0, 2 * np.pi)
        self._phi5 = np.random.uniform(0, 2 * np.pi)

        self.logger.info(
            f"Winding: critical_interval={self.critical_interval}  "
            f"3rd harmonic at {THREE_X_LINE:.0f} Hz  "
            f"5th harmonic at {FIVE_X_LINE:.0f} Hz"
        )

    def _insulation_health(self) -> float:
        """Insulation health index: 1.0 (new) → 0.0 (failed)."""
        return float(max(1.0 - self.interval_count / self.critical_interval, 0.0))

    def _severity(self) -> float:
        return 1.0 - self._insulation_health()

    def _pd_rate(self) -> float:
        """Per-sample probability of a partial discharge event."""
        sev = self._severity()
        return 0.0003 + 0.008 * sev   # 0.03% → 0.83%

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        """
        Magnetic force imbalance from shorted turns → vibration at 2× line (100 Hz).
        Partial discharge events create high-frequency mechanical transients.
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        base = NOM_ACCEL_MS2 * self.load_factor

        if self.system_failure_state:
            # Full insulation failure: violent electromagnetic discharge
            em_100 = 2.0 * np.sin(2 * np.pi * TWO_X_LINE * t)
            # Dense PD event train
            pd_events = _weibull_pd_events(N, 0.02, cluster_size=5)
            pd_amplitudes = np.random.lognormal(np.log(1.5), 0.4, N) * pd_events
            pd_signal = np.zeros(N)
            for idx in np.where(pd_events)[0]:
                pd_signal += damped_impact(t, t[idx], pd_amplitudes[idx],
                                           omega_n=STRUCT_OMEGA_N * 1.5, zeta=0.1)
            noise = np.random.normal(0, 1.0, N)
            return base * 3.5 + em_100 + pd_signal + noise

        # 2× line vibration from magnetic imbalance (grows with shorted turns)
        em_100 = (0.1 + 1.5 * sev) * np.sin(2 * np.pi * TWO_X_LINE * t)

        # Partial discharge transients
        pd_events = _weibull_pd_events(N, self._pd_rate(), cluster_size=2.5)
        pd_signal = np.zeros(N, dtype=float)
        if pd_events.any():
            pd_amps = np.random.lognormal(np.log(0.5 + 1.5 * sev), 0.35, N) * pd_events
            for idx in np.where(pd_events)[0]:
                pd_signal += damped_impact(t, t[idx], pd_amps[idx],
                                           omega_n=STRUCT_OMEGA_N, zeta=STRUCT_ZETA)

        noise = np.random.normal(0, base * 0.04 + 0.15 * sev, N)
        data = base + em_100 + pd_signal + noise

        if self.interval_count >= self.critical_interval and not self.system_failure_state:
            self.system_failure_state = True
            self.logger.warning(f"⚡ WINDING FAILURE at interval {self.interval_count}!")

        return data

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        """
        Correct 50 Hz harmonic model:
          3rd harmonic (150 Hz): dominant, grows quadratically with shorted turns
          5th harmonic (250 Hz): secondary, ~60% of 3rd harmonic
          Current waveform skewness increases (asymmetric due to phase imbalance)
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        fla = NOM_CURRENT_A * self.load_factor

        if self.system_failure_state:
            # Phase-to-ground short: large 3rd/5th harmonics + DC offset
            base = fla * 3.5
            h3 = fla * 1.2 * np.sin(2 * np.pi * THREE_X_LINE * t + self._phi3)
            h5 = fla * 0.7 * np.sin(2 * np.pi * FIVE_X_LINE * t + self._phi5)
            noise = np.random.normal(0, 4.0, N)
            return np.clip(base + h3 + h5 + noise, 1, 150)

        # Progressive harmonic injection
        h3_amp = fla * 0.08 * sev ** 2    # quadratic growth with shorted turns
        h5_amp = fla * 0.05 * sev ** 2    # 5th harmonic ~60% of 3rd

        # Fundamental + harmonics
        i_fund = fla * np.sin(2 * np.pi * 50.0 * t)                        # fundamental
        h3     = h3_amp * np.sin(2 * np.pi * THREE_X_LINE * t + self._phi3)
        h5     = h5_amp * np.sin(2 * np.pi * FIVE_X_LINE  * t + self._phi5)

        # Partial discharge current spikes (short, high-frequency)
        pd_events = _weibull_pd_events(N, self._pd_rate() * 5, cluster_size=3)
        pd_spikes = np.random.lognormal(np.log(0.3 + 1.5 * sev), 0.4, N) * pd_events

        noise = np.random.normal(0, fla * 0.02, N)
        return np.clip(fla + i_fund * 0.0 + h3 + h5 + pd_spikes + noise, 1, 120)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        """
        Partial discharge crackling + growing 100 Hz electromagnetic hum.
        Each PD event sounds like a quiet snap/pop in the early stages,
        growing to loud buzzing and crackling at critical.
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()

        base = NOM_AUDIO_DB * self.load_factor ** 0.05

        if self.system_failure_state:
            # Loud electrical arcing: 100 Hz buzz + crackling
            arc_buzz = 22.0 * np.sin(2 * np.pi * TWO_X_LINE * t)
            crackling = np.random.exponential(4.0, N)   # exponential: skewed crackling
            noise = np.random.normal(0, 5.0, N)
            return base + 14 + arc_buzz + crackling + noise

        # Growing 100 Hz hum (CORRECT: 2× 50 Hz, not 2× 60 Hz)
        em_hum = (2.0 + 10.0 * sev) * np.sin(2 * np.pi * TWO_X_LINE * t)

        # PD crackling events
        pd_events = _weibull_pd_events(N, self._pd_rate() * 3, cluster_size=2)
        pd_crackle = np.random.lognormal(np.log(1.0 + 3.0 * sev), 0.5, N) * pd_events

        noise = np.random.normal(0, 1.0 + 2.5 * sev, N)
        return base + em_hum + pd_crackle + noise


if __name__ == '__main__':
    generator = MotorWindingFailureGenerator()
    generator.run_indefinitely()
