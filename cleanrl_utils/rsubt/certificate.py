"""
RsubtCertificate: Stability certificate with EWMA smoothing and hysteresis.

Combines directional shock and gap ratio into the Rsubt score,
applies exponential weighted moving average, and implements
a hysteresis state machine for alarm states.
"""

from __future__ import annotations

from enum import IntEnum


class AlarmState(IntEnum):
    """Alarm state levels."""
    GREEN = 0
    YELLOW = 1
    RED = 2


class RsubtCertificate:
    """
    Stability certificate with smoothing and hysteresis state machine.
    
    Computes Rsubt = shock / max(gap_ratio, g_min), applies EWMA smoothing,
    and maintains alarm state with hysteresis for robust alerting.
    
    Args:
        g_min: Minimum gap ratio (prevents division by zero/instability).
        ewma_alpha: EWMA smoothing factor (higher = more weight on recent).
        tau_yellow: Threshold for entering yellow state.
        tau_red: Threshold for entering red state.
        yellow_consec: Consecutive ticks needed to enter yellow.
        red_consec: Consecutive ticks needed to enter red.
        clear_consec: Consecutive ticks below clear threshold to clear alarm.
        clear_ratio: Ratio of tau_yellow for clearing (e.g., 0.8).
    """
    
    def __init__(
        self,
        g_min: float = 0.01,
        ewma_alpha: float = 0.1,
        tau_yellow: float = 0.5,
        tau_red: float = 1.0,
        yellow_consec: int = 3,
        red_consec: int = 2,
        clear_consec: int = 5,
        clear_ratio: float = 0.8,
    ):
        self.g_min = g_min
        self.ewma_alpha = ewma_alpha
        self.tau_yellow = tau_yellow
        self.tau_red = tau_red
        self.yellow_consec = yellow_consec
        self.red_consec = red_consec
        self.clear_consec = clear_consec
        self.clear_ratio = clear_ratio
        
        # State
        self.ewma: float = 0.0
        self.raw: float = 0.0
        self.state = AlarmState.GREEN
        
        # Counters for hysteresis
        self._above_yellow_count = 0
        self._above_red_count = 0
        self._below_clear_count = 0
    
    def update(self, shock: float, gap_ratio: float) -> dict:
        """
        Update certificate with new shock and gap measurements.
        
        Args:
            shock: Directional shock α_t(v) in [0, 1].
            gap_ratio: g_t = min_gap / (λ_1 + ε).
        
        Returns:
            Dictionary with raw, ewma, state, shock, gap_ratio.
        """
        # Compute raw Rsubt score
        effective_gap = max(gap_ratio, self.g_min)
        self.raw = shock / effective_gap
        
        # EWMA update
        self.ewma = self.ewma_alpha * self.raw + (1 - self.ewma_alpha) * self.ewma
        
        # Update hysteresis counters and state
        self._update_state()
        
        return {
            "raw": self.raw,
            "ewma": self.ewma,
            "state": self.state,
            "state_int": int(self.state),
            "shock": shock,
            "gap_ratio": gap_ratio,
        }
    
    def _update_state(self) -> None:
        """Update alarm state based on EWMA and hysteresis rules."""
        clear_threshold = self.clear_ratio * self.tau_yellow
        
        # Count consecutive periods above/below thresholds
        if self.ewma > self.tau_red:
            self._above_red_count += 1
            self._above_yellow_count += 1
            self._below_clear_count = 0
        elif self.ewma > self.tau_yellow:
            self._above_red_count = 0
            self._above_yellow_count += 1
            self._below_clear_count = 0
        elif self.ewma < clear_threshold:
            self._above_red_count = 0
            self._above_yellow_count = 0
            self._below_clear_count += 1
        else:
            self._above_red_count = 0
            self._above_yellow_count = 0
            self._below_clear_count = 0
        
        # State transitions
        if self.state == AlarmState.GREEN:
            if self._above_red_count >= self.red_consec:
                self.state = AlarmState.RED
            elif self._above_yellow_count >= self.yellow_consec:
                self.state = AlarmState.YELLOW
        
        elif self.state == AlarmState.YELLOW:
            if self._above_red_count >= self.red_consec:
                self.state = AlarmState.RED
            elif self._below_clear_count >= self.clear_consec:
                self.state = AlarmState.GREEN
        
        elif self.state == AlarmState.RED:
            if self._below_clear_count >= self.clear_consec:
                self.state = AlarmState.GREEN
    
    def reset(self) -> None:
        """Reset certificate state."""
        self.ewma = 0.0
        self.raw = 0.0
        self.state = AlarmState.GREEN
        self._above_yellow_count = 0
        self._above_red_count = 0
        self._below_clear_count = 0
    
    def set_thresholds(self, tau_yellow: float, tau_red: float) -> None:
        """Update alarm thresholds (e.g., after calibration)."""
        self.tau_yellow = tau_yellow
        self.tau_red = tau_red
