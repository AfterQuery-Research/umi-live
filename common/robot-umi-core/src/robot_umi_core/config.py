"""Fixed two-client UMI configuration and dataset-schema derivation.

Each recorder snapshots its one-arm config into the uploaded episode. The server merges
the native ``umi1`` leader and ``umi2`` follower snapshots before processing. Public
collection is always bimanual; one-arm configs exist only as the two transport halves.

``derive()`` is the single source of the LeRobot schema: state/action dims and names,
video keys, gripper keys. The device never touches ``state_dim``; the server never
re-implements naming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_umi_core import layout, ports

CONFIG_VERSION = 1
CONFIG_ENV_VAR = "ROBOT_UMI_CONFIG"

# Per-UMI pose state: [x, y, z, rx, ry, rz] (metres + rotation-vector radians).
STATE_DIM_PER_UMI = 6
STATE_NAMES_PER_UMI = ["x", "y", "z", "rx", "ry", "rz"]

LEADER_UMI = "umi1"
FOLLOWER_UMI = "umi2"
BIMANUAL_UMIS = (LEADER_UMI, FOLLOWER_UMI)
CLIENT_ROLES = {LEADER_UMI: "leader", FOLLOWER_UMI: "follower"}
GRIPPER_TAGS = {LEADER_UMI: (0, 1), FOLLOWER_UMI: (2, 3)}


class ConfigError(ValueError):
    """Malformed or self-inconsistent rig configuration."""


@dataclass(frozen=True)
class CameraConfig:
    device: str  # /dev/v4l/by-path/... (survives USB re-enumeration)
    width: int
    height: int
    fps: int
    calib: str  # per-UMI fisheye .npz, relative to the repo/config root
    exposure: int | None = None


@dataclass(frozen=True)
class LidarConfig:
    iface: str = "eth0"
    host_ip: str = ports.ROBOSENSE_DEFAULT_HOST_IP
    lidar_ip: str = ports.ROBOSENSE_DEFAULT_LIDAR_IP
    # Optional manual override of the per-unit IMU->lidar factory extrinsic, GLIM TUM
    # layout [x, y, z, qx, qy, qz, qw]. Left None, the server auto-extracts it from the
    # lidar's own DIFOP packet (falling back to a hardcoded default); set it only when a
    # unit's DIFOP can't be read. See robot_umi_core.difop.
    t_lidar_imu: tuple[float, ...] | None = None


@dataclass(frozen=True)
class GripperConfig:
    tag_ids: tuple[int, int]  # this arm's ArUco pair (DICT_4X4_50); umi1=(0,1), umi2=(2,3)
    stride: int = 2  # detect every Nth frame
    roi_top_frac: float = 0.0


@dataclass(frozen=True)
class ButtonsConfig:
    """BCM GPIO pins for the record and quit buttons."""

    record: int = 12
    quit: int = 16


@dataclass(frozen=True)
class DisplayConfig:
    """Status LCD: kind + ST7789 control pins (SPI0 data/clk/cs are fixed by the bus)."""

    kind: str = "lcd"  # lcd | none
    dc: int = 25
    rst: int = 27
    bl: int = 18


@dataclass(frozen=True)
class ControlsConfig:
    """Per-Pi HMI wiring: one button set + one LCD per Pi, independent of UMI count."""

    buttons: ButtonsConfig = field(default_factory=ButtonsConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


@dataclass(frozen=True)
class AlignmentConfig:
    """Optional cross-lidar map alignment for bimanual rigs. DISABLED by default: with
    ``enabled=False`` each lidar is SLAM'd independently and the dataset consumes the
    per-lidar GLIM trajectories directly (historical behavior — existing configs, which
    have no ``alignment:`` block, keep working unchanged). When enabled, a post-GLIM
    ``align`` stage registers the two lidar maps into ONE common frame so cross-arm
    geometry is metric.

    ``reference`` names the UMI whose frame is the common frame (None => the first UMI).
    ``tier`` 1 = a single rigid map-to-map registration; 2 = a joint pose graph
    (cross-agent constraints) that also refines each individual trajectory."""

    enabled: bool = False
    reference: str | None = None  # UMI name; None => the first UMI
    tier: int = 1  # 1 = rigid registration, 2 = joint pose graph
    gicp_voxel: float = 0.25  # metres; registration downsample resolution
    # Optional [xmin,ymin,zmin,xmax,ymax,zmax] workspace crop applied to each map before
    # matching, to exclude the moving arms from the registration.
    workspace_bbox: tuple[float, ...] | None = None
    cross_agent_constraints: int = 8  # tier 2: # of cross-agent submap matches to add
    robust_threshold: float = 0.1  # tier 2: GNC/Huber inlier scale (metres)


@dataclass(frozen=True)
class UmiConfig:
    name: str
    camera: CameraConfig
    lidar: LidarConfig
    gripper: GripperConfig | None = None


@dataclass(frozen=True)
class RigConfig:
    umis: tuple[UmiConfig, ...]
    task: str = "umi demo task"
    controls: ControlsConfig = field(default_factory=ControlsConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    version: int = CONFIG_VERSION

    @property
    def n_umis(self) -> int:
        return len(self.umis)


# --- loading -------------------------------------------------------------------------


def load_rig_config(path: str | Path) -> RigConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return parse_rig_config(raw, source=str(path))


def _reject_unknown(mapping: dict, allowed: set[str], where: str, source: str) -> None:
    """Fail on keys we don't recognize. The rig yaml is copied verbatim into every
    episode snapshot, so a stray/typo'd key is both a
    silent misconfiguration and a way for a secret to leak — reject it up front."""
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise ConfigError(f"{source}: unknown key(s) in {where}: {extra}")


def parse_rig_config(raw: Any, source: str = "<config>") -> RigConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    _reject_unknown(
        raw,
        {"version", "umis", "task", "controls", "alignment"},
        "top level",
        source,
    )
    version = raw.get("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        raise ConfigError(f"{source}: unsupported config version {version}")

    umis_raw = raw.get("umis")
    if not isinstance(umis_raw, dict) or not umis_raw:
        raise ConfigError(f"{source}: 'umis' must be a non-empty mapping of name -> config")

    umis: list[UmiConfig] = []
    for name, u in umis_raw.items():
        if not isinstance(u, dict):
            raise ConfigError(f"{source}: umis.{name} must be a mapping")
        umis.append(_parse_umi(name, u, source))

    cfg = RigConfig(
        umis=tuple(umis),
        task=str(raw.get("task", "umi demo task")),
        controls=_parse_controls(raw.get("controls"), source),
        alignment=_parse_alignment(raw.get("alignment"), source),
        version=version,
    )
    _validate(cfg, source)
    return cfg


def _parse_controls(raw: Any, source: str) -> ControlsConfig:
    """Optional top-level `controls:` (button + LCD pins). Omitted keys keep the
    historical hardcoded defaults, so pre-`controls` configs are unchanged."""
    if raw is None:
        return ControlsConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: 'controls' must be a mapping")
    _reject_unknown(raw, {"buttons", "display"}, "controls", source)

    b = raw.get("buttons") or {}
    if not isinstance(b, dict):
        raise ConfigError(f"{source}: controls.buttons must be a mapping")
    _reject_unknown(b, {"record", "quit"}, "controls.buttons", source)
    buttons = ButtonsConfig(
        record=int(b.get("record", 12)),
        quit=int(b.get("quit", 16)),
    )

    d = raw.get("display") or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{source}: controls.display must be a mapping")
    _reject_unknown(d, {"kind", "dc", "rst", "bl"}, "controls.display", source)
    kind = str(d.get("kind", "lcd"))
    if kind not in ("lcd", "none"):
        raise ConfigError(f"{source}: controls.display.kind {kind!r} not in ('lcd', 'none')")
    display = DisplayConfig(
        kind=kind,
        dc=int(d.get("dc", 25)),
        rst=int(d.get("rst", 27)),
        bl=int(d.get("bl", 18)),
    )
    return ControlsConfig(buttons=buttons, display=display)


def _parse_alignment(raw: Any, source: str) -> AlignmentConfig:
    """Optional top-level `alignment:` block. Absent/None => a DISABLED default (each lidar
    SLAM'd independently, dataset reads per-lidar GLIM — the historical path). Unknown keys
    are rejected like everywhere else."""
    if raw is None:
        return AlignmentConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: 'alignment' must be a mapping")
    _reject_unknown(
        raw,
        {
            "enabled",
            "reference",
            "tier",
            "gicp_voxel",
            "workspace_bbox",
            "cross_agent_constraints",
            "robust_threshold",
        },
        "alignment",
        source,
    )
    tier = int(raw.get("tier", 1))
    if tier not in (1, 2):
        raise ConfigError(f"{source}: alignment.tier must be 1 or 2, got {tier}")
    bbox: tuple[float, ...] | None = None
    bbox_raw = raw.get("workspace_bbox")
    if bbox_raw is not None:
        try:
            bbox = tuple(float(v) for v in bbox_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{source}: alignment.workspace_bbox must be a list of numbers"
            ) from exc
        if len(bbox) != 6:
            raise ConfigError(
                f"{source}: alignment.workspace_bbox must be "
                f"[xmin,ymin,zmin,xmax,ymax,zmax], got {len(bbox)}"
            )
    ref = raw.get("reference")
    return AlignmentConfig(
        enabled=bool(raw.get("enabled", False)),
        reference=str(ref) if ref is not None else None,
        tier=tier,
        gicp_voxel=float(raw.get("gicp_voxel", 0.25)),
        workspace_bbox=bbox,
        cross_agent_constraints=int(raw.get("cross_agent_constraints", 8)),
        robust_threshold=float(raw.get("robust_threshold", 0.1)),
    )


def _parse_t_lidar_imu(raw: Any, name: str, source: str) -> tuple[float, ...] | None:
    """Optional per-lidar extrinsic override: 7 finite floats [x,y,z,qx,qy,qz,qw] with a
    non-degenerate quaternion. Fails loudly."""
    if raw is None:
        return None
    try:
        vals = tuple(float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{source}: umis.{name}.lidar.t_lidar_imu must be a list of numbers"
        ) from exc
    if len(vals) != 7:
        raise ConfigError(
            f"{source}: umis.{name}.lidar.t_lidar_imu must have 7 values "
            f"[x,y,z,qx,qy,qz,qw], got {len(vals)}"
        )
    if not all(math.isfinite(v) for v in vals):
        raise ConfigError(f"{source}: umis.{name}.lidar.t_lidar_imu has non-finite values")
    if math.sqrt(sum(v * v for v in vals[3:])) < 1e-6:
        raise ConfigError(f"{source}: umis.{name}.lidar.t_lidar_imu quaternion is degenerate")
    return vals


def _parse_umi(name: str, u: dict, source: str) -> UmiConfig:
    _reject_unknown(u, {"camera", "lidar", "gripper"}, f"umis.{name}", source)
    cam = u.get("camera")
    if not isinstance(cam, dict):
        raise ConfigError(f"{source}: umis.{name}.camera is required")
    _reject_unknown(
        cam,
        {"device", "width", "height", "fps", "calib", "exposure"},
        f"umis.{name}.camera",
        source,
    )
    camera = CameraConfig(
        device=str(cam["device"]),
        width=int(cam["width"]),
        height=int(cam["height"]),
        fps=int(cam["fps"]),
        calib=str(cam["calib"]),
        exposure=int(cam["exposure"]) if "exposure" in cam else None,
    )

    ld = u.get("lidar")
    if not isinstance(ld, dict):
        raise ConfigError(f"{source}: umis.{name}.lidar is required")
    _reject_unknown(
        ld,
        {"iface", "host_ip", "lidar_ip", "t_lidar_imu"},
        f"umis.{name}.lidar",
        source,
    )
    lidar = LidarConfig(
        iface=str(ld.get("iface", "eth0")),
        host_ip=str(ld.get("host_ip", ports.ROBOSENSE_DEFAULT_HOST_IP)),
        lidar_ip=str(ld.get("lidar_ip", ports.ROBOSENSE_DEFAULT_LIDAR_IP)),
        t_lidar_imu=_parse_t_lidar_imu(ld.get("t_lidar_imu"), name, source),
    )

    gripper = None
    if "gripper" in u and u["gripper"] is not None:
        g = u["gripper"]
        _reject_unknown(
            g, {"tag_ids", "stride", "roi_top_frac"}, f"umis.{name}.gripper", source
        )
        tag_ids = tuple(int(t) for t in g["tag_ids"])
        if len(tag_ids) != 2:
            raise ConfigError(f"{source}: umis.{name}.gripper.tag_ids must have 2 ids")
        gripper = GripperConfig(
            tag_ids=tag_ids,  # type: ignore[arg-type]
            stride=int(g.get("stride", 2)),
            roi_top_frac=float(g.get("roi_top_frac", 0.0)),
        )

    return UmiConfig(name=name, camera=camera, lidar=lidar, gripper=gripper)


def _validate(cfg: RigConfig, source: str) -> None:
    names = [u.name for u in cfg.umis]
    if len(set(names)) != len(names):
        raise ConfigError(f"{source}: duplicate UMI names {names}")
    # An explicit alignment reference must name a real UMI.
    if cfg.alignment.reference is not None and cfg.alignment.reference not in names:
        raise ConfigError(
            f"{source}: alignment.reference {cfg.alignment.reference!r} "
            f"is not a UMI name {names}"
        )
    # 4:3 is load-bearing: gripper ArUco tags ride the bottom band; 16:9 clips them.
    for u in cfg.umis:
        if u.gripper is not None and u.camera.width * 3 != u.camera.height * 4:
            raise ConfigError(
                f"{source}: umis.{u.name} has a gripper but camera is "
                f"{u.camera.width}x{u.camera.height} (not 4:3) — tags would be clipped"
            )
    # Gripper tag pairs must be disjoint across arms (a camera glimpsing the other
    # gripper must not cross-detect).
    all_tags = [t for u in cfg.umis if u.gripper for t in u.gripper.tag_ids]
    if len(set(all_tags)) != len(all_tags):
        raise ConfigError(f"{source}: gripper tag_ids overlap across UMIs: {all_tags}")
    # GPIO pins must be distinct: buttons + active LCD control pins share one header,
    # and a collision silently steals a button or bricks the LCD.
    c = cfg.controls
    pins = [
        ("controls.buttons.record", c.buttons.record),
        ("controls.buttons.quit", c.buttons.quit),
    ]
    if c.display.kind == "lcd":
        pins += [
            ("controls.display.dc", c.display.dc),
            ("controls.display.rst", c.display.rst),
            ("controls.display.bl", c.display.bl),
        ]
    seen: dict[int, str] = {}
    for label, pin in pins:
        if pin in seen:
            raise ConfigError(
                f"{source}: GPIO pin {pin} used by both {seen[pin]} and {label}"
            )
        seen[pin] = label


# --- schema + capture-plan derivation -------------------------------------------------


@dataclass(frozen=True)
class DerivedSchema:
    """Everything downstream derives from the rig config — computed in exactly one place."""

    n_umis: int
    state_dim: int
    state_names: list[str]
    video_keys: dict[str, str]  # umi name -> LeRobot video feature key
    gripper_keys: dict[str, str]  # umi name -> gripper width feature key
    fps: int  # dataset timeline fps = first UMI's camera fps
    features: dict[str, dict] = field(default_factory=dict)  # full LeRobot feature dict


def derive(cfg: RigConfig) -> DerivedSchema:
    n = cfg.n_umis
    state_dim = STATE_DIM_PER_UMI * n
    # Each client snapshot contains exactly one fixed arm name.
    state_names = [f"{u.name}_{s}" for u in cfg.umis for s in STATE_NAMES_PER_UMI]
    video_keys = {u.name: f"observation.images.{u.name}" for u in cfg.umis}
    gripper_keys = {
        u.name: f"observation.gripper_width.{u.name}" for u in cfg.umis if u.gripper
    }
    fps = cfg.umis[0].camera.fps

    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"d_{n}" for n in state_names],
        },
    }
    for u in cfg.umis:
        features[video_keys[u.name]] = {
            "dtype": "video",
            "shape": (u.camera.height, u.camera.width, 3),
            "names": ["height", "width", "channels"],
        }
    for name, key in gripper_keys.items():
        features[key] = {"dtype": "float32", "shape": (1,), "names": [f"{name}_width_mm"]}

    return DerivedSchema(
        n_umis=n,
        state_dim=state_dim,
        state_names=state_names,
        video_keys=video_keys,
        gripper_keys=gripper_keys,
        fps=fps,
        features=features,
    )


def client_role(cfg: RigConfig) -> str:
    """Return the fixed role of a one-arm recorder config."""
    names = tuple(u.name for u in cfg.umis)
    if len(names) != 1 or names[0] not in CLIENT_ROLES:
        raise ConfigError(
            f"client config must contain exactly one of {LEADER_UMI!r} or "
            f"{FOLLOWER_UMI!r}, got {list(names)}"
        )
    umi = cfg.umis[0]
    expected_tags = GRIPPER_TAGS[umi.name]
    if umi.gripper is None or umi.gripper.tag_ids != expected_tags:
        raise ConfigError(
            f"{umi.name} must use ArUco tags {list(expected_tags)}"
        )
    return CLIENT_ROLES[umi.name]


def require_bimanual(cfg: RigConfig) -> DerivedSchema:
    """Validate the only dataset contract: ordered umi1/umi2 with 12-D state/action."""
    names = tuple(u.name for u in cfg.umis)
    if names != BIMANUAL_UMIS:
        raise ConfigError(
            f"dataset config must be paired umi1/umi2, got {list(names)}"
        )
    for umi in cfg.umis:
        expected_tags = GRIPPER_TAGS[umi.name]
        if umi.gripper is None or umi.gripper.tag_ids != expected_tags:
            raise ConfigError(
                f"{umi.name} must use ArUco tags {list(expected_tags)}"
            )
    derived = derive(cfg)
    if derived.state_dim != 12:
        raise ConfigError(f"bimanual dataset state must be 12-D, got {derived.state_dim}")
    return derived


def capture_plan(cfg: RigConfig) -> list[tuple[str, str]]:
    """The fixed lidar-plus-camera capture plan, in start order."""
    plan: list[tuple[str, str]] = []
    for u in cfg.umis:
        plan.append((layout.sensor_key(u.name, layout.KIND_LIDAR), layout.KIND_LIDAR))
        plan.append((layout.sensor_key(u.name, layout.KIND_CAMERA), layout.KIND_CAMERA))
    return plan


def referenced_blobs(cfg: RigConfig) -> list[str]:
    """Config-relative paths of every calib blob the config references (for snapshotting)."""
    return [u.camera.calib for u in cfg.umis]
