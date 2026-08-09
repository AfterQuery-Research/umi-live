from __future__ import annotations

from PIL import Image
from umi_flow_device.display import StatusDisplay


class FakePanel:
    width = 320
    height = 240

    def init(self) -> None: ...
    def show(self, image: Image.Image) -> None: ...
    def close(self) -> None: ...


def test_check_failure_is_persistent_and_cleared_when_recording_starts() -> None:
    display = StatusDisplay(driver=FakePanel())
    display.notify_check_failure(("umi1 tags missing: 2", "umi2 tags missing: 0,1"))

    assert display._check_failure == ("umi1 tags missing: 2", "umi2 tags missing: 0,1")
    assert display._compose().getpixel((5, 40)) == (92, 20, 24)

    display.set_recording(True)
    assert display._check_failure == ()
