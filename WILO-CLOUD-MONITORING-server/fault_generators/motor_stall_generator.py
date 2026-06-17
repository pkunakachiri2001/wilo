"""
Motor Stall Generator — SUDDEN FAULT (locked-rotor event)

Physics (CORRECTED):
  During a stall the rotor stops rotating completely.
  • Vibration DROPS — no rotation means no rotational excitation.
    Only remaining vibration: electromagnetic hum from the stalled flux (100 Hz, 200 Hz).
  • Current SPIKES to locked-rotor current (LRC ≈ 6–8× FLA).
    As the rotor heats up, resistance increases and LRC slowly decays.
  • Audio: high-pitched electromagnetic hum replaces normal motor noise.

Pre-fault baseline is flat — Motor Stall is a SUDDEN fault.
Event manager uses failure_idx directly (no gradual onset scan needed).
"""

import numpy as np
from .base_generator import (
    BaseGenerator,
    NOM_CURRENT_A, NOM_ACCEL_MS2, NOM_AUDIO_DB,
    TWO_X_LINE, SHAFT_FREQ,
)


class MotorStallGenerator(BaseGenerator):
    """Motor Stall — sudden locked-rotor fault."""

    def __init__(self):
        super().__init__('Motor Stall')
        self.is_healthy = True
        self.spike_interval = np.random.randint(10, 16)
        self.logger.info(f"Stall will trigger at interval {self.spike_interval}")

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------
    def generate_acceleration_data(self):
        """
        Pre-stall: steady rotational vibration at 1× shaft.
        At stall: rotor stops → rotational vibration collapses to near-zero.
                  Only electromagnetic hum remains (100/200 Hz).
        Post-stall (system_failure): same collapsed vibration, slowly decaying.
        """
        t = self.generate_time_array()
        N = len(t)

        if self.system_failure_state:
            # Rotor stationary: only 100 Hz electromagnetic hum + very low noise
            em_hum  = 0.08 * np.sin(2 * np.pi * TWO_X_LINE * t)
            em_hum2 = 0.04 * np.sin(2 * np.pi * 200 * t)
            noise   = np.random.normal(0, 0.03, N)
            return em_hum + em_hum2 + noise

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            # THE STALL MOMENT:
            #   Transition interval — first half shows normal vibration
            #   collapsing as rotor decelerates to zero.
            self.failure_triggered = True
            self.system_failure_state = True
            self.is_healthy = False
            self.logger.warning(f"⚡ STALL at interval {self.interval_count}! "
                                f"Rotor decelerating to zero.")

            # Rotor deceleration: vibration amplitude decays exponentially
            # as shaft frequency chirps down from nominal → 0
            decel_tau = 0.3   # seconds — time constant of speed decay
            freq_decay = SHAFT_FREQ * np.exp(-t / decel_tau)    # chirping shaft freq
            vib_decay  = NOM_ACCEL_MS2 * self.load_factor * np.exp(-t / decel_tau)
            rotational = vib_decay * np.sin(2 * np.pi * np.cumsum(freq_decay) / SHAFT_FREQ)
            em_hum = 0.08 * (1 - np.exp(-t / decel_tau)) * np.sin(2 * np.pi * TWO_X_LINE * t)
            noise  = np.random.normal(0, 0.03, N)
            return rotational + em_hum + noise

        # Healthy baseline: 1× shaft rotation + small noise
        one_x = NOM_ACCEL_MS2 * self.load_factor * np.sin(2 * np.pi * SHAFT_FREQ * t)
        noise = np.random.normal(0, NOM_ACCEL_MS2 * self.load_factor * 0.04, N)
        return one_x + noise

    # ------------------------------------------------------------------
    # Current
    # ------------------------------------------------------------------
    def generate_current_data(self):
        """
        Pre-stall:  nominal current.
        At stall:   locked-rotor current (LRC) = 6–8× FLA, then decays
                    as rotor temperature rises (resistance increases → LRC drops).
        Post-stall: sustained high current ~4–5× FLA (motor still energised).
        """
        t = self.generate_time_array()
        N = len(t)
        fla = NOM_CURRENT_A * self.load_factor

        if self.system_failure_state:
            # Sustained locked-rotor: ~4–5× FLA after thermal equilibrium
            lrc_sustained = fla * 4.5
            noise = np.random.normal(0, 2.5, N)
            return np.clip(lrc_sustained + noise, fla * 2, fla * 8)

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            # Immediate locked-rotor spike: 6–8× FLA, decaying slightly
            lrc_peak = fla * np.random.uniform(6.0, 8.0)
            lrc_decay = lrc_peak * np.exp(-t * 0.5)   # thermal resistance rise
            noise = np.random.normal(0, 3.0, N)
            return np.clip(lrc_decay + noise, fla, fla * 9)

        # Normal operating current
        noise = np.random.normal(0, fla * 0.03, N)
        return np.clip(fla + noise, 1, 100)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def generate_audio_data(self):
        """
        Pre-stall:  normal motor hum.
        At stall:   high-pitched electromagnetic squeal (rotor locked in
                    AC flux → strong 100 Hz hum + harmonics).
        Post-stall: loud 100/200 Hz hum replaces normal rotation noise.
        """
        t = self.generate_time_array()
        N = len(t)

        if self.system_failure_state:
            # Electromagnetic hum dominates — loud and tonal
            hum_100  = 15.0 * np.sin(2 * np.pi * TWO_X_LINE * t)
            hum_200  = 8.0  * np.sin(2 * np.pi * 200 * t)
            noise    = np.random.normal(0, 2.5, N)
            return NOM_AUDIO_DB + 6 + hum_100 + hum_200 + noise

        if self.interval_count == self.spike_interval and not self.failure_triggered:
            # Stall onset: sudden change from rotational noise to EM hum
            squeal  = 18.0 * np.sin(2 * np.pi * TWO_X_LINE * t)
            harmonic = 9.0 * np.sin(2 * np.pi * 200 * t)
            noise   = np.random.normal(0, 2.0, N)
            return NOM_AUDIO_DB + 8 + squeal + harmonic + noise

        # Normal: rotational hum at shaft frequency
        shaft_hum = 4.0 * np.sin(2 * np.pi * SHAFT_FREQ * t)
        noise = np.random.normal(0, 1.5, N)
        return NOM_AUDIO_DB * self.load_factor ** 0.05 + shaft_hum + noise


if __name__ == '__main__':
    generator = MotorStallGenerator()
    generator.run_indefinitely()
