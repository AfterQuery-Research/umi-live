"""LCD status monitor (Waveshare 2" ST7789, 240x320, SPI0, DC=25/RST=4/BL=18).

Shows lidar presence and temperature, REC/IDLE, pending episode count, and the last
upload result (transient banner on success, persistent on failure).

`StatusDisplay` keeps lock-guarded state updated from several threads (monitor
poller, App loop, uploader) and a background render thread redraws the LCD.
`ST7789` is the panel driver; spidev/gpiozero are imported lazily so this module
imports fine off-Pi (CI). `NullDisplay` is the no-op used without hardware.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib import import_module
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

from umi_flow_device.lidar_monitor import LidarHealth

UPLOAD_PROCESSING = "processing"
UPLOAD_UPLOADING = "uploading"
UPLOAD_OK = "ok"
UPLOAD_FAIL = "fail"


class Display(ABC):
    """Status-monitor interface. Methods are called from multiple threads."""

    @abstractmethod
    def set_lidar_health(self, umi: str, health: LidarHealth) -> None: ...

    @abstractmethod
    def set_recording(self, recording: bool) -> None: ...

    @abstractmethod
    def set_episodes(self, count: int) -> None: ...

    @abstractmethod
    def notify_upload(self, state: str) -> None: ...

    @abstractmethod
    def set_sending(self, episode_id: str | None) -> None: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class NullDisplay(Display):
    """No-op display: used when no LCD is attached (the default off-Pi)."""

    def set_lidar_health(self, umi: str, health: LidarHealth) -> None: ...
    def set_recording(self, recording: bool) -> None: ...
    def set_episodes(self, count: int) -> None: ...
    def notify_upload(self, state: str) -> None: ...
    def set_sending(self, episode_id: str | None) -> None: ...
    def start(self) -> None: ...
    def close(self) -> None: ...


class Panel(Protocol):
    """Structural interface for a panel driver (ST7789 or a test fake)."""

    width: int
    height: int

    def init(self) -> None: ...
    def show(self, image: Image.Image) -> None: ...
    def close(self) -> None: ...


def image_to_rgb565_be(image: Image.Image) -> bytes:
    """Pack an (H, W, 3) RGB image into ST7789 RGB565, big-endian."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return rgb565.astype(">u2").tobytes()


class ST7789:
    """Waveshare 2" ST7789 over SPI0 + gpiozero control pins (proven on the rig)."""

    _NATIVE_W = 240
    _NATIVE_H = 320
    _CHUNK = 4096

    def __init__(
        self,
        spi_bus: int = 0,
        spi_dev: int = 0,
        speed_hz: int = 8_000_000,
        dc: int = 25,
        rst: int = 4,
        bl: int = 18,
        rotation: int = 90,
    ):
        from gpiozero import DigitalOutputDevice  # ty: ignore[unresolved-import]

        spidev = import_module("spidev")
        if rotation not in (0, 90, 180, 270):
            raise ValueError(f"rotation must be 0/90/180/270, got {rotation}")
        self.rotation = rotation
        if rotation in (90, 270):
            self.width, self.height = self._NATIVE_H, self._NATIVE_W
        else:
            self.width, self.height = self._NATIVE_W, self._NATIVE_H
        self._dc = DigitalOutputDevice(dc)
        self._rst = DigitalOutputDevice(rst)
        self._bl = DigitalOutputDevice(bl)
        self._spi = spidev.SpiDev()
        self._spi.open(spi_bus, spi_dev)
        self._spi.max_speed_hz = speed_hz
        self._spi.mode = 0

    def _cmd(self, c: int) -> None:
        self._dc.off()
        self._spi.writebytes([c])

    def _data(self, buf: bytes) -> None:
        self._dc.on()
        for i in range(0, len(buf), self._CHUNK):
            self._spi.writebytes2(buf[i : i + self._CHUNK])

    def init(self) -> None:
        self._bl.on()
        self._rst.on()
        time.sleep(0.05)
        self._rst.off()
        time.sleep(0.05)
        self._rst.on()
        time.sleep(0.15)
        self._cmd(0x01)  # SWRESET
        time.sleep(0.15)
        self._cmd(0x11)  # SLPOUT
        time.sleep(0.12)
        self._cmd(0x3A)
        self._data(bytes([0x55]))  # COLMOD 16-bit/px
        self._cmd(0x36)
        self._data(bytes([0x00]))  # MADCTL
        self._cmd(0x21)  # INVON (IPS panel)
        self._cmd(0x13)  # NORON
        self._cmd(0x29)  # DISPON
        time.sleep(0.1)

    def show(self, image: Image.Image) -> None:
        if self.rotation:
            image = image.rotate(self.rotation, expand=True)
        self._cmd(0x2A)
        self._data(bytes([0, 0, (self._NATIVE_W - 1) >> 8, (self._NATIVE_W - 1) & 0xFF]))
        self._cmd(0x2B)
        self._data(bytes([0, 0, (self._NATIVE_H - 1) >> 8, (self._NATIVE_H - 1) & 0xFF]))
        self._cmd(0x2C)  # RAMWR
        self._data(image_to_rgb565_be(image))

    def close(self) -> None:
        try:
            self._bl.off()
        finally:
            self._dc.close()
            self._rst.close()
            self._bl.close()
            self._spi.close()


_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_font_cache: dict[int, Any] = {}


def _font(size: int) -> Any:
    if size not in _font_cache:
        from PIL import ImageFont

        font = None
        for path in _FONT_PATHS:
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        _font_cache[size] = font if font is not None else ImageFont.load_default()
    return _font_cache[size]


_BG = (14, 17, 22)
_FG = (220, 226, 232)
_DIM = (120, 130, 140)
_GREEN = (54, 224, 122)
_RED = (235, 70, 70)
_AMBER = (240, 190, 60)
_BLUE = (90, 150, 240)


class StatusDisplay(Display):
    """Aggregates rig status and renders it to the LCD on a background thread."""

    def __init__(
        self,
        driver: Panel | None = None,
        notify_s: float = 3.0,
        fps: float = 4.0,
        now: Callable[[], float] = time.monotonic,
        show_status: bool = True,
        title: str = "UMI Flow",
    ):
        # show_status=False renders a lidar-health-only screen (no REC/IDLE box, no
        # episode count, no upload banner) — used on a follower, whose operator drives
        # recording from the leader and only needs its lidar's liveness + temperature.
        self._show_status = show_status
        self._title = title  # top-left header (e.g. "Leader"/"Follower")
        self._driver: Panel = driver if driver is not None else ST7789()
        self._notify_s = notify_s
        self._period = 1.0 / fps
        self._now = now
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lidars: dict[str, LidarHealth] = {}
        self._recording = False
        self._episodes = 0
        self._upload_state: str | None = None
        self._upload_ts = 0.0
        self._sending: str | None = None  # episode streaming to the server right now

    # --- writers -----------------------------------------------------------
    def set_lidar_health(self, umi: str, health: LidarHealth) -> None:
        with self._lock:
            self._lidars[umi] = health

    def set_recording(self, recording: bool) -> None:
        with self._lock:
            self._recording = recording

    def set_episodes(self, count: int) -> None:
        with self._lock:
            self._episodes = count

    def notify_upload(self, state: str) -> None:
        with self._lock:
            self._upload_state = state
            self._upload_ts = self._now()

    def set_sending(self, episode_id: str | None) -> None:
        """Episode whose tar is streaming to the server right now (None = idle).
        Drives the live SENDING banner; the pending count shows how many remain."""
        with self._lock:
            self._sending = episode_id

    # --- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._driver.init()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self._driver.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._driver.show(self._compose())
            except Exception as e:  # never let a render error kill the thread
                print(f"display render error: {e}")
            self._stop.wait(self._period)

    # --- rendering -----------------------------------------------------------
    def _upload_banner(self, now: float) -> tuple[str, tuple[int, int, int]] | None:
        if self._upload_state is None:
            return None
        if self._upload_state == UPLOAD_PROCESSING:
            return "processing...", _AMBER
        if self._upload_state == UPLOAD_UPLOADING:
            return "uploading...", _AMBER
        if self._upload_state == UPLOAD_OK:
            if now - self._upload_ts > self._notify_s:
                return None
            return "uploaded OK", _GREEN
        if self._upload_state == UPLOAD_FAIL:
            return "upload failed", _RED
        return None

    def _compose(self) -> Image.Image:
        now = self._now()
        with self._lock:
            lidars = dict(self._lidars)
            recording = self._recording
            episodes = self._episodes
            banner = self._upload_banner(now)
            sending = self._sending

        w, h = self._driver.width, self._driver.height  # landscape: 320 x 240
        img = Image.new("RGB", (w, h), _BG)
        d = ImageDraw.Draw(img)
        d.text((10, 4), self._title, fill=_BLUE, font=_font(18))

        # one row per configured lidar: dot + name + type + temperature
        y = 36
        if not lidars:
            d.text((16, y), "no lidar configured", fill=_DIM, font=_font(16))
            y += 28
        for umi, health in sorted(lidars.items()):
            ok = health.present
            d.ellipse([16, y + 2, 32, y + 18], fill=_GREEN if ok else _RED)
            label = f"{umi} {health.lidar_type or 'lidar'}" if ok else f"{umi} no lidar"
            d.text((42, y), label, fill=_FG if ok else _DIM, font=_font(16))
            if ok and health.temp_c is not None:
                temp = health.temp_c
                temp_color = _GREEN if temp < 60 else (_AMBER if temp < 75 else _RED)
                d.text((228, y), f"{temp:4.1f}\N{DEGREE SIGN}C", fill=temp_color, font=_font(16))
            y += 28

        # recording box (left) + episode count (right) + upload banner: only on a
        # full-status display (leader). A follower shows lidar health only.
        if self._show_status:
            top = max(y + 6, 118)
            d.rectangle([14, top, 156, top + 66], fill=_RED if recording else (30, 34, 40))
            d.text((40, top + 20), "REC" if recording else "IDLE", fill=_FG, font=_font(26))
            d.text((176, top + 12), "episodes", fill=_DIM, font=_font(16))
            d.text((176, top + 32), str(episodes), fill=_FG, font=_font(26))

            # SENDING takes the banner while a tar is actively streaming (serial upload);
            # the count shows how many episodes are still pending. Falls back to the
            # transient upload-result banner (uploaded OK / failed) when idle.
            if sending is not None:
                d.rectangle([0, h - 34, w, h], fill=_AMBER)
                plural = "episode" if episodes == 1 else "episodes"
                d.text(
                    (16, h - 28),
                    f"\N{UPWARDS ARROW} sending  ({episodes} {plural})",
                    fill=(0, 0, 0),
                    font=_font(18),
                )
            elif banner is not None:
                text, color = banner
                d.rectangle([0, h - 34, w, h], fill=color)
                d.text((16, h - 28), text, fill=(0, 0, 0), font=_font(18))
        return img
