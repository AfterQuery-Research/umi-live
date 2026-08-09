"""Button input: emits TOGGLE / CANCEL / QUIT events.

Two physical Pi buttons (BCM pin -> ground; gpiozero's pull-up handles idle):
- record (default 12): tap = TOGGLE (start/stop), hold = CANCEL
- quit   (default 16): hold = QUIT

The debounce logic is battle-tested from the previous rig: contact bounce on release
fires press+release twice within ~10 ms, which without guards turns one tap into
stop-then-restart, and a hold-for-cancel's release into a spurious restart.
"""

from __future__ import annotations

import select
import sys
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from queue import Empty, Queue


class Event(Enum):
    """A button press."""

    START = auto()
    STOP = auto()
    TOGGLE = auto()
    CANCEL = auto()
    QUIT = auto()


class Buttons(ABC):
    @abstractmethod
    def poll(self) -> Event | None:
        """Return the next event, or None if no button was pressed."""


class KeyboardButtons(Buttons):
    """Dev buttons: type a key then Enter. s=start e=stop c=cancel q=quit.

    POSIX-only (select() on stdin) — fine on Mac and the Pi.
    """

    KEYMAP = {
        "s": Event.START,
        "e": Event.STOP,
        "c": Event.CANCEL,
        "q": Event.QUIT,
    }

    def poll(self) -> Event | None:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return None
        line = sys.stdin.readline()
        if line == "":  # stdin closed
            return Event.QUIT
        return self.KEYMAP.get(line.strip().lower())


class GpioButtons(Buttons):
    """Two physical Pi buttons (record / quit), dual-debounced."""

    def __init__(
        self,
        record_pin: int = 17,
        quit_pin: int = 27,
        hold_s: float = 2.0,
        bounce_s: float = 0.05,
        toggle_debounce_s: float = 0.4,
    ):
        from gpiozero import Button  # ty: ignore[unresolved-import]

        self._events: Queue[Event] = Queue()
        self._record_held = False
        # Ignore a second TOGGLE within this window: contact bounce on release fires
        # two toggles per tap (stop -> immediately start again). A human can't toggle
        # this fast intentionally (they'd hold for cancel).
        self._toggle_debounce_s = toggle_debounce_s
        self._last_toggle = 0.0
        # When a hold-for-cancel is released, contact bounce re-fires press+release;
        # the bounce press clears _record_held, so the bounce release would emit a
        # spurious TOGGLE and restart recording right after a cancel. Suppress toggles
        # for the debounce window after any hold-release.
        self._hold_released_at = 0.0
        self._record = Button(record_pin, bounce_time=bounce_s, hold_time=hold_s)
        self._quit = Button(quit_pin, hold_time=hold_s)
        self._record.when_pressed = self._on_record_pressed
        self._record.when_held = self._on_record_held
        self._record.when_released = self._on_record_released
        self._quit.when_held = lambda: self._events.put(Event.QUIT)

    def _on_record_pressed(self) -> None:
        self._record_held = False

    def _on_record_held(self) -> None:
        self._record_held = True
        self._events.put(Event.CANCEL)

    def _on_record_released(self) -> None:
        if self._record_held:
            self._hold_released_at = time.monotonic()  # arm the post-cancel guard
            return
        now = time.monotonic()
        if now - self._hold_released_at < self._toggle_debounce_s:
            return  # bounce right after a hold-for-cancel: don't restart recording
        if now - self._last_toggle < self._toggle_debounce_s:
            return  # contact bounce / accidental double-tap
        self._last_toggle = now
        self._events.put(Event.TOGGLE)

    def poll(self) -> Event | None:
        try:
            return self._events.get_nowait()
        except Empty:
            return None

    def close(self) -> None:
        self._record.close()
        self._quit.close()


class ScriptedButtons(Buttons):
    """Plays back a fixed sequence of events (tests / headless self-checks)."""

    def __init__(self, events: list[Event | None]):
        self.events = list(events)

    def poll(self) -> Event | None:
        if not self.events:
            return Event.QUIT
        return self.events.pop(0)
