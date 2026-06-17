"""
Motor Electrical Fault Generator — GRADUAL FAULT (phase imbalance / line-to-ground fault)

Physics (CORRECTED — 50 Hz system):
  Phase imbalance and line-to-ground faults create:
    • Sidebands in current at f_line ± 2·f_slip  (slip sidebands)
    • Growing 2× line (100 Hz) vibration from magnetic pull imbalance
    • Sporadic high-amplitude arc transients in current (damped, not Poisson)
    • Current spectrum: fundamental + sidebands growing with severity

  CORRECTED frequencies (50 Hz system):
    • Line fundamental: 50 Hz
    • 2× line:         100 Hz  (was incorrectly 120 Hz in old code)
    • Slip sidebands: 50 ± 2 × f_slip  (f_slip = slip × 50 Hz, typically 2–5 Hz)
    • 3× line:         150 Hz (odd harmonic from phase imbalance)

  Arc transients modelled as damped sinusoidal impulses (NOT Poisson integers).
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    damped_impact,
    TWO_X_LINE, THREE_X_LINE,
    SHAFT_FREQ, LINE_FREQ,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
    STRUCT_OMEGA_N, STRUCT_ZETA,
)


class MotorElectricalFaultGenerator(BaseGenerator):
    """Motor Electrical Fault — phase imbalance with correct 50 Hz harmonic model."""

    def __init__(self):
        super().__init__('Motor Electrical Fault')
        self.critical_interval = np.random.randint(11, 16)

        # Slip frequency (random per run — realistic slip varies with load)
        slip_pct = np.random.uniform(0.02, 0.06)     # 2–6% slip
        self._f_slip = slip_pct * LINE_FREQ           # ~1–3 Hz

        # Phase offset (electrical fault has a fixed but unknown phase reference)
        self._phi_fault = np.random.uniform(0, 2 * np.pi)

        self.logger.info(
            f"Electrical fault: critical_interval={self.critical_interval}  "
            f"f_slip={self._f_slip:.2f} Hz  "
            f"sidebands at {LINE_FREQ - 2*self._f_slip:.1f} / {LINE_FREQ + 2*self._f_slip:.1f} Hz"
        )

    def _severity(self) -> float:
        return float(np.clip(self.interval_count / self.critical_interval, 0.0, 1.0))

    def _arc_rate(self) -> float:
        """Per-sample probability of an electrical arc transient."""
        sev = self._severity()
        return 0.001 + 0.05 * sev    # 0.1% → 5.1%

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        """
        Magnetic pull imbalance from phase fault → vibration at 2× line (100 Hz).
        Arc transients create mechanical shock (damped sinusoidal, NOT Poisson).
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        base = NOM_ACCEL_MS2 * self.load_factor

        if self.system_failure_state:
            # Catastrophic fault: strong 100 Hz magnetic pull + arc shocks
            em_pull = 3.5 * np.sin(2 * np.pi * TWO_X_LINE * t)
            arc_events = np.random.binomial(1, 0.025, N).astype(bool)
            arc_signal = np.zeros(N, dtype=float)
            for idx in np.where(arc_events)[0]:
                A_arc = np.random.lognormal(np.log(2.0), 0.4)
                arc_signal += damped_impact(t, t[idx], A_arc,
                                            omega_n=STRUCT_OMEGA_N * 1.2, zeta=0.09)
            noise = np.random.normal(0, 1.2, N)
            return base * 4 + em_pull + arc_signal + noise

        # Growing 100 Hz magnetic pull imbalance
        em_100 = (0.08 + 1.8 * sev) * np.sin(2 * np.pi * TWO_X_LINE * t + self._phi_fault)

        # Sporadic arc transients (damped impulse, not Poisson integer)
        arc_events = np.random.binomial(1, self._arc_rate(), N).astype(bool)
        arc_signal = np.zeros(N, dtype=float)
        for idx in np.where(arc_events)[0]:
            A_arc = np.random.lognormal(np.log(0.4 + 1.5 * sev), 0.35)
            arc_signal += damped_impact(t, t[idx], A_arc, omega_n=STRUCT_OMEGA_N, zeta=STRUCT_ZETA)

        noise = np.random.normal(0, base * 0.05 + 0.2 * sev, N)
        data = base + em_100 + arc_signal + noise

        if self.interval_count >= self.critical_interval and not self.system_failure_state:
            self.system_failure_state = True
            self.logger.warning(f"⚡ ELECTRICAL FAULT at interval {self.interval_count}!")

        return data

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        """
        Phase imbalance signature in current:
          • Slip sidebands: f_line ± 2·f_slip
          • Growing 3rd harmonic (150 Hz) from phase imbalance
          • Sporadic high-amplitude arc transients
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()
        fla = NOM_CURRENT_A * self.load_factor

        if self.system_failure_state:
            # Fault current: high DC + strong harmonics + arc spikes
            base = fla * 3.8
            h3 = fla * 0.9 * np.sin(2 * np.pi * THREE_X_LINE * t)
            arc_spikes = np.random.binomial(1, 0.03, N) * np.random.lognormal(
                np.log(fla * 0.8), 0.4, N)
            noise = np.random.normal(0, 5.0, N)
            return np.clip(base + h3 + arc_spikes + noise, 1, 200)

        # Slip sidebands in current spectrum
        f_lower = LINE_FREQ - 2 * self._f_slip
        f_upper = LINE_FREQ + 2 * self._f_slip
        sideband_amp = fla * 0.04 * sev    # grows with severity

        sideband = (sideband_amp * np.sin(2 * np.pi * f_lower * t + self._phi_fault) +
                    sideband_amp * np.sin(2 * np.pi * f_upper * t))

        # 3rd harmonic from phase imbalance
        h3 = fla * 0.05 * sev * np.sin(2 * np.pi * THREE_X_LINE * t + self._phi_fault)

        # Arc current spikes (damped transients)
        arc_events = np.random.binomial(1, self._arc_rate() * 4, N).astype(bool)
        arc_current = np.zeros(N, dtype=float)
        for idx in np.where(arc_events)[0]:
            A_arc = np.random.lognormal(np.log(fla * 0.3 + fla * sev), 0.5)
            arc_current[idx] += A_arc  # current spikes are fast (sub-sample), not ringing

        dc_rise = fla * (1.0 + 0.12 * sev)   # slight DC rise from phase imbalance load
        noise = np.random.normal(0, fla * 0.02, N)
        return np.clip(dc_rise + sideband + h3 + arc_current + noise, 1, 150)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        """
        100 Hz electrical hum (CORRECT for 50 Hz system) + crackling arc sounds.
        Growing 100 Hz hum is the primary audible signature.
        """
        t = self.generate_time_array()
        N = len(t)
        sev = self._severity()

        base = NOM_AUDIO_DB * self.load_factor ** 0.05

        if self.system_failure_state:
            # Loud electrical fault: 100 Hz buzz + crackling
            hum_100 = 24.0 * np.sin(2 * np.pi * TWO_X_LINE * t)
            crackle = np.random.exponential(5.0, N)   # skewed crackling amplitude
            noise = np.random.normal(0, 5.0, N)
            return base + 12 + hum_100 + crackle + noise

        # Growing 100 Hz hum (CORRECTED from old 120 Hz)
        hum_100 = (1.5 + 10.0 * sev) * np.sin(2 * np.pi * TWO_X_LINE * t + self._phi_fault)
        # Sporadic arc crackling
        arc_events = np.random.binomial(1, self._arc_rate() * 3, N).astype(bool)
        crackle = np.random.exponential(1.0 + 4.0 * sev, N) * arc_events
        noise = np.random.normal(0, 1.5 + 2.0 * sev, N)
        return base + hum_100 + crackle + noise


if __name__ == '__main__':
    generator = MotorElectricalFaultGenerator()
    generator.run_indefinitely()
