"""
Pump Impeller Damage Generator — SUDDEN FAULT (hydraulic imbalance)

Physics (CORRECTED — BPF was 9 Hz in old code; should be ~60.8 Hz):

  For Wilo pump (5-vane impeller, 730 RPM motor):
    f_shaft = 730/60 ≈ 12.17 Hz
    BPF = N_vanes × f_shaft = 5 × 12.17 ≈ 60.8 Hz

  Impeller damage (broken/eroded vane) signatures:
    • Lost vane symmetry → hydraulic unbalance at 1× shaft (12.17 Hz)
    • Amplitude and phase modulated BPF vibration (one vane missing or deformed)
    • Sidebands at BPF ± f_shaft, BPF ± 2·f_shaft (amplitude modulation by shaft rotation)
    • Current: torque ripple at BPF frequency (hydraulic forces modulate load)
    • Audio: rhythmic "thudding" at shaft frequency + broadband grinding

  Pre-damage: smooth BPF ripple from normal vane pressure pulses.
  At damage:  sudden hydraulic imbalance — amplitude modulated BPF + 1× shaft spike.
  Post-damage: sustained imbalance with increasing severity from erosion.
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    impact_train, damped_impact,
    BPF, SHAFT_FREQ,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
    STRUCT_OMEGA_N, STRUCT_ZETA,
)

PUMP_OMEGA_N = 2 * np.pi * 2000
PUMP_ZETA    = 0.05


class PumpImpellerDamageGenerator(BaseGenerator):
    """Pump Impeller Damage — sudden hydraulic imbalance with correct BPF model."""

    def __init__(self):
        super().__init__('Pump Impeller Damage')
        self.is_healthy = True
        self.spike_interval = np.random.randint(10, 16)
        # Random initial phase for shaft rotation
        self._phi_shaft = np.random.uniform(0, 2 * np.pi)
        self.logger.info(f"Impeller damage at interval {self.spike_interval}  "
                         f"BPF={BPF:.1f} Hz  1×shaft={SHAFT_FREQ:.2f} Hz")

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        """
        Damage = missing/deformed vane → amplitude-modulated BPF.
        Sidebands appear at BPF ± f_shaft, ± 2·f_shaft.
        """
        t = self.generate_time_array()
        N = len(t)

        if self.system_failure_state:
            # Sustained imbalance: 1× shaft (hydraulic unbalance) + amplitude-mod BPF
            one_x   = 2.0 * self.load_factor * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi_shaft)
            # Amplitude modulation: BPF carrier × shaft-rate envelope
            envelope = 1.0 + 0.8 * np.abs(np.sin(np.pi * SHAFT_FREQ * t))
            bpf_mod  = impact_train(t, BPF, 2.2 * self.load_factor,
                                    amplitude_sigma=0.4, omega_n=PUMP_OMEGA_N, zeta=PUMP_ZETA)
            # Sidebands via AM: BPF ± shaft
            sideband_lo = 0.8 * self.load_factor * np.sin(2 * np.pi * (BPF - SHAFT_FREQ) * t)
            sideband_hi = 0.8 * self.load_factor * np.sin(2 * np.pi * (BPF + SHAFT_FREQ) * t)
            noise = np.random.normal(0, 0.5, N)
            return NOM_ACCEL_MS2 * self.load_factor + one_x + bpf_mod * envelope + \
                   sideband_lo + sideband_hi + noise

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            self.failure_triggered = True
            self.system_failure_state = True
            self.is_healthy = False
            self.logger.warning(f"⚙️ IMPELLER DAMAGE at interval {self.interval_count}! "
                                f"Hydraulic unbalance onset — BPF={BPF:.1f} Hz")

            # Damage onset: sudden hydraulic imbalance impulse + BPF modulation
            onset_impact = damped_impact(t, 0.01, 5.0 * self.load_factor,
                                          omega_n=PUMP_OMEGA_N, zeta=PUMP_ZETA)
            one_x = 1.8 * self.load_factor * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi_shaft)
            bpf_train = impact_train(t, BPF, 1.8 * self.load_factor,
                                     amplitude_sigma=0.4, omega_n=PUMP_OMEGA_N, zeta=PUMP_ZETA)
            noise = np.random.normal(0, 0.4, N)
            return NOM_ACCEL_MS2 * self.load_factor + onset_impact + one_x + bpf_train + noise

        # Normal: smooth BPF pressure ripple from all 5 vanes (balanced)
        bpf_ripple  = 0.2 * self.load_factor * np.sin(2 * np.pi * BPF * t)
        shaft_ripple = 0.12 * self.load_factor * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi_shaft)
        noise = np.random.normal(0, NOM_ACCEL_MS2 * self.load_factor * 0.05, N)
        return NOM_ACCEL_MS2 * self.load_factor + bpf_ripple + shaft_ripple + noise

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        """
        Hydraulic torque ripple at BPF frequency → current oscillation at BPF.
        Also: 1× shaft torque oscillation from hydraulic imbalance.
        """
        t = self.generate_time_array()
        N = len(t)
        fla = NOM_CURRENT_A * self.load_factor

        if self.system_failure_state:
            # Strong BPF current ripple + elevated mean from hydraulic inefficiency
            base = fla * 2.0
            bpf_ripple  = fla * 0.35 * np.sin(2 * np.pi * BPF * t)
            shaft_ripple = fla * 0.20 * np.sin(2 * np.pi * SHAFT_FREQ * t + self._phi_shaft)
            noise = np.random.normal(0, 2.5, N)
            return np.clip(base + bpf_ripple + shaft_ripple + noise, fla * 0.5, fla * 5)

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            # Sudden load surge from hydraulic imbalance
            base = fla * 3.2
            bpf_ripple = fla * 0.5 * np.sin(2 * np.pi * BPF * t)
            noise = np.random.normal(0, 4.0, N)
            return np.clip(base + bpf_ripple + noise, fla, fla * 6)

        # Normal: steady with small BPF torque ripple
        bpf_ripple = fla * 0.03 * np.sin(2 * np.pi * BPF * t)
        noise = np.random.normal(0, fla * 0.025, N)
        return np.clip(fla + bpf_ripple + noise, 1, 100)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        """
        Impeller damage: rhythmic "thudding" at 1× shaft from hydraulic kick
        at each revolution + BPF grinding from the damaged vane.
        """
        t = self.generate_time_array()
        N = len(t)

        if self.system_failure_state:
            base = NOM_AUDIO_DB + 10
            # Thudding at shaft frequency (hydraulic unbalance)
            thud = 12.0 * np.abs(np.sin(np.pi * SHAFT_FREQ * t + self._phi_shaft))
            # BPF grinding
            grind_train = impact_train(t, BPF, 5.0, amplitude_sigma=0.45,
                                       omega_n=2 * np.pi * 600, zeta=0.18)
            noise = np.random.normal(0, 4.0, N)
            return base + thud + grind_train + noise

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            base = NOM_AUDIO_DB + 14
            crack = damped_impact(t, 0.005, 18.0, omega_n=2 * np.pi * 800, zeta=0.15)
            bpf_grind = impact_train(t, BPF, 6.0, amplitude_sigma=0.5,
                                     omega_n=2 * np.pi * 600, zeta=0.2)
            noise = np.random.normal(0, 4.0, N)
            return base + crack + bpf_grind + noise

        # Normal: BPF pressure tone
        bpf_tone = 2.0 * self.load_factor * np.sin(2 * np.pi * BPF * t)
        noise = np.random.normal(0, 1.5, N)
        return NOM_AUDIO_DB * self.load_factor ** 0.05 + bpf_tone + noise


if __name__ == '__main__':
    generator = PumpImpellerDamageGenerator()
    generator.run_indefinitely()
