"""MJCF assembly helpers for MjDex robot assets."""

from __future__ import annotations

from pathlib import Path
import math

from dm_control import mjcf

from mjdex.transform import (
    _cross,
    _normalize,
    quat_mul,
    x_tilt_quat,
    y_tilt_quat,
    yaw_quat,
)


MjDex_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = MjDex_ROOT / "assets"

ROBOT_SCENES = {
    "ur5e": ASSET_ROOT / "universal_robots_ur5e" / "scene.xml",
    "fr3": ASSET_ROOT / "franka_fr3" / "scene.xml",
}

ROBOT_MODELS = {
    "ur5e": ASSET_ROOT / "universal_robots_ur5e" / "ur5e.xml",
    "fr3": ASSET_ROOT / "franka_fr3" / "fr3.xml",
}

GRIPPER_MODELS = {
    "robotiq_2f85": ASSET_ROOT / "robotiq_2f85_v4" / "2f85.xml",
    "inspire_left": ASSET_ROOT / "inspire_rh56f1" / "left_hand.xml",
    "inspire_right": ASSET_ROOT / "inspire_rh56f1" / "right_hand.xml",
    "allegro_left": ASSET_ROOT / "wonik_allegro" / "left_hand.xml",
    "allegro_right": ASSET_ROOT / "wonik_allegro" / "right_hand.xml",
    "sharpa_left": ASSET_ROOT / "sharpa_wave" / "left_hand.xml",
    "sharpa_right": ASSET_ROOT / "sharpa_wave" / "right_hand.xml",
}
ALLEGRO_ATTACHMENT_POS = (0.0, 0.0091, 0.1)
ALLEGRO_ATTACHMENT_QUAT = (
    0.5,
    -0.5,
    -0.5,
    -0.5,
)
ALLEGRO_ADAPTOR_RADIUS = 0.03773072
ALLEGRO_ADAPTOR_HALF_HEIGHT = 0.005
ALLEGRO_ADAPTOR_OVERLAP = 0.001
ALLEGRO_ADAPTOR_POS = (
    0.0,
    0.0,
    ALLEGRO_ADAPTOR_HALF_HEIGHT - ALLEGRO_ADAPTOR_OVERLAP,
)

DUALDEX_TABLE_SIZE = (1.30, 0.80, 0.75)
DEFAULT_ROBOT_BASE_POS = (0.0, 0.0, 0.0)
DEFAULT_FLOOR_Z = -DUALDEX_TABLE_SIZE[2]
TABLE_OVERHANG_BY_ROBOT = {
    "ur5e": 0.10,
    "fr3": 0.125,
}
DUAL_ARM_TABLE_OVERHANG_BY_ROBOT = {
    "fr3": 0.0,
    "ur5e": 0.0,
}
ROBOT_LOCAL_BASE_QUAT = {
    "fr3": (1.0, 0.0, 0.0, 0.0),
    "ur5e": (0.0, 0.0, 0.0, -1.0),
}
RENDER_CAMERA_NAME = "render_camera"
RENDER_CAMERA_POS = (1.55, 0.0, 0.6)
RENDER_CAMERA_TARGET = (0.25, 0.0, 0.16)
DUAL_RENDER_CAMERA_POS = (1.35, -1.35, 0.95)
DUAL_RENDER_CAMERA_TARGET = (0.15, 0.0, 0.22)
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720


def load_mjcf(path: str | Path) -> mjcf.RootElement:
    """Load an MJCF file with escaped names for safe attachment."""

    return mjcf.from_path(Path(path).as_posix(), escape_separators=True)


def build_robot_scene(
    robot: str,
    base_pos: tuple[float, float, float] = DEFAULT_ROBOT_BASE_POS,
    base_quat: tuple[float, float, float, float] | None = None,
) -> mjcf.RootElement:
    """Load a robot scene by short name."""

    try:
        scene_path = ROBOT_SCENES[robot]
    except KeyError as exc:
        raise ValueError(
            f"Unknown robot '{robot}'. Options: {sorted(ROBOT_SCENES)}"
        ) from exc
    scene = load_mjcf(scene_path)
    set_robot_base_pose(scene, pos=base_pos, quat=base_quat)
    set_offscreen_render_size(scene)
    return scene


def build_robot_model(
    robot: str,
    base_pos: tuple[float, float, float] = DEFAULT_ROBOT_BASE_POS,
    base_quat: tuple[float, float, float, float] | None = None,
) -> mjcf.RootElement:
    """Load a robot-only MJCF by short name."""

    try:
        robot_path = ROBOT_MODELS[robot]
    except KeyError as exc:
        raise ValueError(
            f"Unknown robot '{robot}'. Options: {sorted(ROBOT_MODELS)}"
        ) from exc
    model = load_mjcf(robot_path)
    set_robot_base_pose(model, pos=base_pos, quat=base_quat)
    return model


def set_robot_base_pose(
    model: mjcf.RootElement,
    pos: tuple[float, float, float] = DEFAULT_ROBOT_BASE_POS,
    quat: tuple[float, float, float, float] | None = None,
) -> None:
    """Place the root robot base body in world coordinates."""

    base = model.find("body", "base")
    if base is None:
        raise ValueError("Could not find robot root body 'base'.")
    base.pos = pos
    if quat is not None:
        base.quat = quat


def set_offscreen_render_size(
    model: mjcf.RootElement,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
) -> None:
    """Set MuJoCo's offscreen framebuffer size for RGB rendering."""

    visual_global = getattr(model.visual, "global")
    visual_global.offwidth = int(width)
    visual_global.offheight = int(height)


def camera_xyaxes_from_lookat(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Return MuJoCo camera xyaxes for a camera at ``pos`` looking at ``target``."""

    forward = _normalize(tuple(target[i] - pos[i] for i in range(3)))
    camera_z = tuple(-value for value in forward)
    world_up = (0.0, 0.0, 1.0)
    camera_x = _normalize(_cross(world_up, camera_z))
    camera_y = _cross(camera_z, camera_x)
    return (*camera_x, *camera_y)


def add_render_camera(
    scene: mjcf.RootElement,
    pos: tuple[float, float, float] = RENDER_CAMERA_POS,
    target: tuple[float, float, float] = RENDER_CAMERA_TARGET,
) -> mjcf.Element:
    """Add or update the default fixed render camera."""

    camera = scene.find("camera", RENDER_CAMERA_NAME)
    xyaxes = camera_xyaxes_from_lookat(pos, target)
    if camera is None:
        return scene.worldbody.add(
            "camera",
            name=RENDER_CAMERA_NAME,
            pos=pos,
            xyaxes=xyaxes,
            fovy=55,
        )
    camera.pos = pos
    camera.xyaxes = xyaxes
    camera.fovy = 55
    return camera


def mounted_robot_base_quat(
    robot: str,
    mount_quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return the world base quat produced by attaching a robot to a mount site."""

    return tuple(
        quat_mul(mount_quat, ROBOT_LOCAL_BASE_QUAT.get(robot, (1.0, 0.0, 0.0, 0.0)))
    )


def set_floor_height(model: mjcf.RootElement, z: float = DEFAULT_FLOOR_Z) -> None:
    """Place the visual/collision floor below the robot-base frame."""

    floor = model.find("geom", "floor")
    if floor is not None:
        floor.pos = (0.0, 0.0, z)


def build_gripper(gripper: str) -> mjcf.RootElement:
    """Load a gripper model by short name."""

    try:
        gripper_path = GRIPPER_MODELS[gripper]
    except KeyError as exc:
        raise ValueError(
            f"Unknown gripper '{gripper}'. Options: {sorted(GRIPPER_MODELS)}"
        ) from exc
    return load_mjcf(gripper_path)


def build_allegro_mount(gripper_model: mjcf.RootElement) -> mjcf.RootElement:
    """Build a DualDex-style adaptor body and attach Allegro below it."""

    mount = mjcf.RootElement(model="allegro_mount")
    mount.compiler.angle = "radian"
    adaptor = mount.worldbody.add("body", name="adaptor", pos=ALLEGRO_ADAPTOR_POS)
    adaptor.add(
        "geom",
        name="adaptor",
        type="cylinder",
        size=(ALLEGRO_ADAPTOR_RADIUS, ALLEGRO_ADAPTOR_HALF_HEIGHT),
        rgba=(0.2, 0.2, 0.2, 1.0),
    )
    attachment = adaptor.add(
        "site",
        name="allegro_attachment_frame",
        pos=ALLEGRO_ATTACHMENT_POS,
        quat=ALLEGRO_ATTACHMENT_QUAT,
        size=(0.001,),
        group=4,
        rgba=(0.1, 0.1, 0.1, 0.3),
    )
    attachment.attach(gripper_model)
    return mount


def attach_gripper(
    scene: mjcf.RootElement,
    gripper: str | mjcf.RootElement,
    attachment_site: str = "attachment_site",
) -> mjcf.RootElement:
    """Attach a gripper model to a named site in ``scene``."""

    gripper_model = build_gripper(gripper) if isinstance(gripper, str) else gripper
    site = scene.find("site", attachment_site)
    if site is None:
        raise ValueError(f"Could not find attachment site '{attachment_site}'.")
    if isinstance(gripper, str) and gripper.startswith("allegro_"):
        gripper_model = build_allegro_mount(gripper_model)
    site.attach(gripper_model)
    return scene


def add_dualdex_table(
    scene: mjcf.RootElement,
    pos: tuple[float, float, float] = (0.0, 0.0, DEFAULT_FLOOR_Z),
    rgba: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0),
) -> mjcf.Element:
    """Add a box-based table using DualDex table dimensions.

    ``pos`` is the table footprint center in the robot-base/world frame, with
    ``z`` at floor height. By default, the tabletop surface is at world z=0 so
    the robot base frame stays the absolute coordinate frame.
    The DualDex table dimensions are rotated 90 degrees here: the short side
    runs along world x, and the long side runs along world y.
    """

    length, depth, height = DUALDEX_TABLE_SIZE
    x_size = depth
    y_size = length
    top_thickness = 0.05
    leg_width = 0.05
    leg_height = height - top_thickness

    table = scene.worldbody.add("body", name="table", pos=pos)
    table.add(
        "geom",
        name="tabletop",
        type="box",
        size=(x_size / 2, y_size / 2, top_thickness / 2),
        pos=(0, 0, height - top_thickness / 2),
        rgba=rgba,
        friction=(0.8, 0.01, 0.001),
    )
    table.add(
        "site",
        name="table_top",
        pos=(0, 0, height),
        size=(0.01,),
        rgba=(0.1, 0.1, 0.1, 0.25),
        group=4,
    )

    leg_x = x_size / 2 - leg_width
    leg_y = y_size / 2 - leg_width
    for i, (x, y) in enumerate(
        [
            (-leg_x, -leg_y),
            (-leg_x, leg_y),
            (leg_x, -leg_y),
            (leg_x, leg_y),
        ]
    ):
        table.add(
            "geom",
            name=f"table_leg_{i}",
            type="box",
            size=(leg_width / 2, leg_width / 2, leg_height / 2),
            pos=(x, y, leg_height / 2),
            rgba=rgba,
            friction=(0.8, 0.01, 0.001),
        )

    return table


def default_table_pos_for_robot(robot: str) -> tuple[float, float, float]:
    """Return table pose that supports the robot base without moving the base.

    The table is rotated so its short side runs along world x and long side
    runs left-right. ``TABLE_OVERHANG_BY_ROBOT`` controls how far the tabletop
    extends behind the robot base center.
    """

    overhang = TABLE_OVERHANG_BY_ROBOT.get(robot, 0.10)
    short_side = DUALDEX_TABLE_SIZE[1]
    return (short_side / 2 - overhang, 0.0, DEFAULT_FLOOR_Z)


def default_dual_arm_table_pos_for_robot(robot: str) -> tuple[float, float, float]:
    """Return table pose for dual-arm scenes without changing single-arm offsets."""

    overhang = DUAL_ARM_TABLE_OVERHANG_BY_ROBOT.get(robot, 0.0)
    short_side = DUALDEX_TABLE_SIZE[1]
    return (short_side / 2 - overhang, 0.0, DEFAULT_FLOOR_Z)


def build_single_robot_scene(
    robot: str = "ur5e",
    gripper: str = "robotiq_2f85",
    attachment_site: str = "attachment_site",
    include_table: bool = True,
    table_pos: tuple[float, float, float] | None = None,
    robot_base_pos: tuple[float, float, float] = DEFAULT_ROBOT_BASE_POS,
) -> mjcf.RootElement:
    """Build a Menagerie-style single robot scene with an attached gripper."""

    scene = build_robot_scene(robot, base_pos=robot_base_pos)
    if include_table:
        if table_pos is None:
            table_pos = default_table_pos_for_robot(robot)
        set_floor_height(scene, z=table_pos[2])
        add_dualdex_table(scene, pos=table_pos)
    attach_gripper(scene, gripper, attachment_site=attachment_site)
    add_render_camera(scene)
    return scene


def build_empty_robot_scene() -> mjcf.RootElement:
    """Build a shared MjDex scene without any robot bodies."""

    scene = mjcf.RootElement(model="mjdex scene")
    scene.compiler.angle = "radian"
    scene.visual.headlight.diffuse = (0.55, 0.55, 0.55)
    scene.visual.headlight.ambient = (0.18, 0.18, 0.18)
    scene.visual.headlight.specular = (0.2, 0.2, 0.2)
    scene.visual.rgba.force = (1, 0, 0, 1)
    visual_global = getattr(scene.visual, "global")
    visual_global.azimuth = 120
    visual_global.elevation = -20
    visual_global.offwidth = RENDER_WIDTH
    visual_global.offheight = RENDER_HEIGHT
    scene.visual.map.force = 0.01
    scene.visual.quality.shadowsize = 8192
    scene.asset.add(
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=(0.88, 0.88, 0.88),
        rgb2=(0.98, 0.98, 0.98),
        width=800,
        height=800,
    )
    scene.asset.add(
        "texture",
        type="2d",
        name="groundplane",
        builtin="checker",
        mark="edge",
        rgb1=(0.93, 0.93, 0.93),
        rgb2=(0.78, 0.78, 0.78),
        markrgb=(0.08, 0.08, 0.08),
        width=300,
        height=300,
    )
    scene.asset.add(
        "material",
        name="groundplane",
        texture="groundplane",
        texuniform=True,
        texrepeat=(5, 5),
        reflectance=0,
    )
    scene.worldbody.add(
        "light",
        pos=(0, 0, 3),
        dir=(0, 0, -1),
        directional=True,
        diffuse=(0.45, 0.45, 0.45),
        specular=(0.15, 0.15, 0.15),
    )
    scene.worldbody.add(
        "geom",
        name="floor",
        size=(0, 0, 0.05),
        type="plane",
        material="groundplane",
        pos=(0, 0, DEFAULT_FLOOR_Z),
    )
    scene.worldbody.add(
        "site",
        name="world_origin",
        type="sphere",
        pos=(0, 0, 0),
        size=(0.025,),
        rgba=(1, 0, 0, 0.9),
        group=4,
    )
    return scene


def default_dual_arm_mounts(
    robot: str,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    """Return default floating base poses for a dual-arm setup."""

    if robot == "fr3":
        tilt = math.radians(45)
        return {
            "left": ((0.05, 0.11, 0.1), x_tilt_quat(-tilt)),
            "right": ((0.05, -0.11, 0.1), x_tilt_quat(tilt)),
        }
    if robot == "ur5e":
        forward_mount = y_tilt_quat(math.radians(90))
        return {
            "left": ((-0.15, 0.25, 0.65), forward_mount),
            "right": ((-0.15, -0.25, 0.65), forward_mount),
        }
    raise ValueError(
        f"Dual-arm defaults are not defined for '{robot}'. Options: ['fr3', 'ur5e']."
    )


def build_dual_robot_scene(
    robot: str = "fr3",
    gripper: str | dict[str, str] = "robotiq_2f85",
    include_table: bool = True,
    table_pos: tuple[float, float, float] | None = None,
    mounts: (
        dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]
        | None
    ) = None,
) -> mjcf.RootElement:
    """Build a dual-arm scene with two mounted robot copies."""

    scene = build_empty_robot_scene()
    if include_table:
        if table_pos is None:
            table_pos = default_dual_arm_table_pos_for_robot(robot)
        set_floor_height(scene, z=table_pos[2])
        add_dualdex_table(scene, pos=table_pos)

    if mounts is None:
        mounts = default_dual_arm_mounts(robot)

    for side, (pos, quat) in mounts.items():
        marker_rgba = (0.1, 0.35, 1.0, 0.9) if side == "left" else (1.0, 0.6, 0.0, 0.9)
        scene.worldbody.add(
            "site",
            name=f"{side}_base_origin",
            type="sphere",
            pos=pos,
            size=(0.035,),
            rgba=marker_rgba,
            group=4,
        )
        mount = scene.worldbody.add(
            "site",
            name=f"{side}_mount",
            pos=pos,
            quat=quat,
            group=4,
            size=(0.01,),
            rgba=(0.1, 0.1, 0.1, 0.3),
        )
        robot_model = build_robot_model(robot)
        robot_model.model = f"{side}_{robot}"
        side_gripper = gripper[side] if isinstance(gripper, dict) else gripper
        attach_gripper(robot_model, side_gripper)
        mount.attach(robot_model)

    add_render_camera(
        scene, pos=DUAL_RENDER_CAMERA_POS, target=DUAL_RENDER_CAMERA_TARGET
    )
    return scene
