"""
Pump Cavitation Generator — SUDDEN FAULT (vapour bubble collapse)

Physics:
  Cavitation occurs when local pressure drops below vapour pressure,
  forming bubbles that collapse violently (Rayleigh collapse) when
  reaching higher-pressure zones.

  CORRECTED signal model:
    • Bubble collapses produce clustered impact bursts at blade-pass frequency (BPF ≈ 60.8 Hz)
      — NOT random i.i.d. Poisson counts
    • Each collapse is a damped sinusoidal transient (Rayleigh collapse → structural response)
    • Collapses cluster near BPF because pressure recovery repeats at each vane pass
    • Sub-harmonic at 0.5×BPF appears during developed cavitation (recirculation zone)
    • Current: pump works harder against cavitation head loss → current RISES (not obvious spike)
    • Audio: broadband "popcorn" noise — each pop is a damped impact, not white noise

  Cavitation develops almost immediately (SUDDEN fault type):
    Pre-cavitation: smooth pump operation with regular BPF ripple
    At trigger:     bubble collapse bursts begin — instant onset
    Post-trigger:   sustained high-intensity cavitation erosion
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    impact_train, damped_impact,
    BPF, SHAFT_FREQ,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
    STRUCT_OMEGA_N, STRUCT_ZETA,
)

# Cavitation structural resonance (pump casing — typically stiffer than motor frame)
PUMP_OMEGA_N = 2 * np.pi * 2000    # ~2000 Hz pump casing resonance
PUMP_ZETA    = 0.05                 # light damping (water-filled)


class PumpCavitationGenerator(BaseGenerator):
    """Pump Cavitation — sudden BPF-synchronised bubble collapse model."""

    def __init__(self):
        super().__init__('Pump Cavitation')
        self.is_healthy = True
        self.spike_interval = np.random.randint(10, 16)
        self.logger.info(f"Cavitation will trigger at interval {self.spike_interval}  "
                         f"BPF={BPF:.1f} Hz")

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        """
        Pre-cavitation: smooth BPF ripple (normal pump pressure pulses).
        At cavitation: bubble collapse bursts at BPF — damped sinusoidal impacts.
        Post-cavitation: sustained intense cavitation erosion.
        """
        t = self.generate_time_array()
        N = len(t)

        if self.system_failure_state:
            # Sustained severe cavitation: dense BPF + 2×BPF impact trains
            bpf_hits  = impact_train(t, BPF,       amplitude_mean=3.5 * self.load_factor,
                                     amplitude_sigma=0.5, omega_n=PUMP_OMEGA_N, zeta=PUMP_ZETA)
            bpf2_hits = impact_train(t, 2 * BPF,   amplitude_mean=1.8 * self.load_factor,
                                     amplitude_sigma=0.55, omega_n=PUMP_OMEGA_N, zeta=PUMP_ZETA)
            sub_harm  = impact_train(t, 0.5 * BPF, amplitude_mean=1.2 * self.load_factor,
                                     amplitude_sigma=0.6, omega_n=PUMP_OMEGA_N, zeta=PUMP_ZETA,
                                     jitter_frac=0.12)
            noise = np.random.normal(0, 0.8 * self.load_factor, N)
            return bpf_hits + bpf2_hits + sub_harm + noise

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            self.failure_triggered = True
            self.system_failure_state = True
            self.is_healthy = False
            self.logger.warning(f"💥 CAVITATION onset at interval {self.interval_count}! "
                                f"Bubble collapses at BPF={BPF:.1f} Hz")
            # Onset: sudden large collapse event + BPF train beginning
            onset_pulse = damped_impact(t, 0.02, 5.0 * self.load_factor,
                                        omega_n=PUMP_OMEGA_N, zeta=PUMP_ZETA)
            bpf_hits = impact_train(t, BPF, 2.8 * self.load_factor,
                                    amplitude_sigma=0.45, omega_n=PUMP_OMEGA_N, zeta=PUMP_ZETA)
            noise = np.random.normal(0, 0.6, N)
            return onset_pulse + bpf_hits + noise

        # Normal operation: smooth BPF pressure ripple (NOT random)
        bpf_ripple = 0.3 * self.load_factor * np.sin(2 * np.pi * BPF * t)
        shaft_ripple = 0.15 * self.load_factor * np.sin(2 * np.pi * SHAFT_FREQ * t)
        noise = np.random.normal(0, NOM_ACCEL_MS2 * self.load_factor * 0.06, N)
        return NOM_ACCEL_MS2 * self.load_factor + bpf_ripple + shaft_ripple + noise

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        """
        Cavitation reduces pump head → motor runs at lower load temporarily,
        then works harder against recirculation → current becomes irregular.
        """
        t = self.generate_time_array()
        N = len(t)
        fla = NOM_CURRENT_A * self.load_factor

        if self.system_failure_state:
            # Sustained cavitation: irregular loading with BPF current ripple
            base = fla * 1.8   # elevated from inefficiency
            bpf_ripple = fla * 0.25 * np.sin(2 * np.pi * BPF * t)
            noise = np.random.normal(0, 3.0, N)
            return np.clip(base + bpf_ripple + noise, fla * 0.5, fla * 4)

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            # Onset: brief current drop (head loss) then surge
            drop_then_surge = fla * (1.5 - 0.8 * np.exp(-t / 0.3))
            noise = np.random.normal(0, 4.0, N)
            return np.clip(drop_then_surge + noise, fla * 0.3, fla * 4)

        # Normal: stable current with small BPF ripple
        bpf_ripple = fla * 0.04 * np.sin(2 * np.pi * BPF * t)
        noise = np.random.normal(0, fla * 0.025, N)
        return np.clip(fla + bpf_ripple + noise, 1, 100)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        """
        Pre-cavitation: smooth pump hum at BPF.
        Cavitation: "popcorn" audio — each bubble collapse is a distinct pop
                    modelled as a damped audio impulse (NOT white noise).
        """
        t = self.generate_time_array()
        N = len(t)

        if self.system_failure_state:
            # Loud broadband roar + BPF popping
            base = NOM_AUDIO_DB + 15
            # Popping: dense impact train at BPF (audio domain)
            pop_train = impact_train(t, BPF, 8.0, amplitude_sigma=0.6,
                                     omega_n=2 * np.pi * 800, zeta=0.2,  # audio resonance
                                     jitter_frac=0.1)
            broadband = np.random.normal(0, 6.0, N)
            return base + pop_train + broadband

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            base = NOM_AUDIO_DB + 18
            pop_train = impact_train(t, BPF, 10.0, amplitude_sigma=0.55,
                                     omega_n=2 * np.pi * 800, zeta=0.2)
            broadband = np.random.normal(0, 5.0, N)
            return base + pop_train + broadband

        # Normal: regular BPF pressure pulse tone
        bpf_tone = 2.5 * self.load_factor * np.sin(2 * np.pi * BPF * t)
        noise = np.random.normal(0, 1.5, N)
        return NOM_AUDIO_DB * self.load_factor ** 0.05 + bpf_tone + noise


if __name__ == '__main__':
    generator = PumpCavitationGenerator()
    generator.run_indefinitely()
