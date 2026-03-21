"""
lights.py  —  Phase 6: LED control for QCar 2
Maps current state to correct LED array for QCar.read_write_std().

LED array layout (verified from qcar.py lines 400-415):
  LEDs[0] = left indicators  (front+rear)
  LEDs[1] = left indicators  (group 2)
  LEDs[2] = right indicators (front+rear)
  LEDs[3] = right indicators (group 2)
  LEDs[4] = brake lights     (all 4)
  LEDs[5] = spare
  LEDs[6] = headlights       (front left+right)
  LEDs[7] = headlights       (rear)

Confirmed from lane_following example: [0,0,0,0,0,0,1,1] = headlights only.
"""
import numpy as np

# State constants (must match state_machine.py)
STATE_IDLE        = 'IDLE'
STATE_NAVIGATING  = 'NAVIGATING'
STATE_AVOIDING    = 'AVOIDING'
STATE_WAITING     = 'WAITING'
STATE_STOPPED     = 'STOPPED'
STATE_ARRIVED     = 'ARRIVED'

# Behaviour constants (from obstacle_detector.py)
BEHAVIOUR_NAVIGATE       = 'NAVIGATE'
BEHAVIOUR_WAIT           = 'WAIT'
BEHAVIOUR_AVOID          = 'AVOID'
BEHAVIOUR_EMERGENCY_STOP = 'EMERGENCY_STOP'


class LightController:
    """
    Converts (state, avoid_side) → LEDs[8] numpy array.

    Usage:
        lights = LightController()
        leds = lights.get_leds(state='NAVIGATING', avoid_side='left')
        qcar.read_write_std(throttle, steering, leds)
    """

    def __init__(self):
        self._blink_state = False
        self._blink_tick  = 0
        self._blink_rate  = 5   # toggle every N calls (~0.17s at 30Hz)

    def get_leds(self, state: str, avoid_side: str = 'left') -> np.ndarray:
        """
        Returns LEDs[8] numpy array for the given state.

        Prints a compact LED status line to terminal on every change.
        """
        self._blink_tick += 1
        if self._blink_tick >= self._blink_rate:
            self._blink_state = not self._blink_state
            self._blink_tick  = 0

        blink = 1 if self._blink_state else 0

        if state == STATE_IDLE:
            # All lights off
            leds = [0, 0, 0, 0,  0, 0,  0, 0]

        elif state == STATE_NAVIGATING:
            # Headlights on, everything else off
            leds = [0, 0, 0, 0,  0, 0,  1, 1]

        elif state == STATE_AVOIDING:
            # Headlights + indicator on avoidance side (blinking)
            if avoid_side == 'left':
                leds = [blink, blink, 0, 0,  0, 0,  1, 1]
            else:
                leds = [0, 0, blink, blink,  0, 0,  1, 1]

        elif state == STATE_WAITING:
            # Headlights + both indicators blinking (hazard = waiting for person)
            leds = [blink, blink, blink, blink,  0, 0,  1, 1]

        elif state == STATE_STOPPED:
            # Headlights + brake lights
            leds = [0, 0, 0, 0,  1, 0,  1, 1]

        elif state == STATE_ARRIVED:
            # All lights off — mission complete
            leds = [0, 0, 0, 0,  0, 0,  0, 0]

        else:
            leds = [0, 0, 0, 0,  0, 0,  0, 0]

        return np.array(leds, dtype=np.float64)

    @staticmethod
    def describe(leds: np.ndarray) -> str:
        """Human-readable LED status for terminal/dashboard."""
        parts = []
        if leds[6] or leds[7]:  parts.append("HEAD")
        if leds[4]:              parts.append("BRAKE")
        if leds[0] or leds[1]:  parts.append("L-IND")
        if leds[2] or leds[3]:  parts.append("R-IND")
        return '+'.join(parts) if parts else "OFF"
