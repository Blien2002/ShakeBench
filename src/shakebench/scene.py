"""C2 layout: independent Panda and textured worktable on one vibrating floor."""

from __future__ import annotations

import warp as wp
from newton import BodyFlags
from isaaclab.physics import PhysicsEvent
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from .arena import load_room_arena_cfg
from .config import BenchmarkConfig, YCB_ASSETS
from .panel import control_panel_layout, panel_table_top_z_m
from .panel_controls import (
    PanelConsoleCollisionCfg,
    PanelControlArticulationCfg,
    spawn_panel_console_collision,
    spawn_panel_control_articulation,
)
from .shaker import (
    ShakerBaseVisualCfg,
    ShakerGeometryCfg,
    make_shaker_leg_collection_cfg,
    spawn_shaker_base_visuals,
)
from .wrist_camera import WristCameraAssemblyCfg, spawn_wrist_camera_assembly
from .visual_assets import (
    ControlPanelAppearanceCfg,
    PlatformAppearanceCfg,
    ShallowBinWallsCfg,
    TexturedTableSurfaceCfg,
    WorktableAppearanceCfg,
    spawn_control_panel_appearance,
    spawn_platform_appearance,
    spawn_shallow_bin_walls,
    spawn_textured_table_surface,
    spawn_worktable_appearance,
)


# Isaac Lab 3.0 beta currently points at the not-yet-published Isaac 6.0
# asset bucket on this machine.  The 5.0 bucket is the newest locally verified
# public asset snapshot (HTTP 200 for Panda and YCB).  Keeping
# this root here makes the standalone project immune to the host kit-file bug.
ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac"
ISAACLAB_ASSET_ROOT = f"{ASSET_ROOT}/IsaacLab"


def make_sim_cfg(
    cfg: BenchmarkConfig,
    device: str = "cuda:0",
    solver_iterations: int | None = None,
) -> SimulationCfg:
    """Return a conservative Newton/MJWarp configuration for fast moving contacts."""

    return SimulationCfg(
        dt=cfg.dt,
        device=device,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=cfg.material_mu,
            dynamic_friction=cfg.material_mu,
            restitution=0.0,
        ),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                iterations=solver_iterations or 80,
                ls_iterations=24,
                cone="elliptic",
                impratio=6.0,
                njmax=420,
                nconmax=180,
                ccd_iterations=50,
                update_data_interval=1,
                use_mujoco_contacts=True,
            ),
            num_substeps=cfg.solver_substeps,
            use_cuda_graph=False,
            debug_mode=False,
        ),
    )


def _franka_cfg():
    robot = FRANKA_PANDA_HIGH_PD_CFG.copy()
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.spawn.usd_path = f"{ISAACLAB_ASSET_ROOT}/Robots/FrankaEmika/panda_instanceable.usd"
    robot.spawn.activate_contact_sensors = True
    # The common vibration frame is imposed through the root state.  Making
    # the official fixed-base asset floating exposes those six base DoFs to
    # Newton while retaining the vendor meshes, inertias and joint limits.
    robot.spawn.articulation_props.fix_root_link = False
    robot.spawn.rigid_props.disable_gravity = False
    robot.spawn.rigid_props.max_depenetration_velocity = 2.0
    return robot


@configclass
class BenchmarkSceneCfg(InteractiveSceneCfg):
    """Single or vectorized benchmark worlds."""

    # Room visuals are isolated from the task physics, following the Arena
    # split used by robosuite / LIBERO and the layout-style split in RoboCasa.
    room = AssetBaseCfg(
        # The laboratory shell is global scene context and must not be cloned
        # once per parallel task environment.
        prim_path="/World/RoomArena",
        spawn=load_room_arena_cfg(),
    )

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.035)),
    )

    # The visible common vibration floor.  Robot and worktable are mounted on
    # it independently; neither is placed on top of the other.
    platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/VibrationFloor",
        spawn=sim_utils.CuboidCfg(
            size=(1.60, 1.10, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=2.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0010, rest_offset=0.0001),
            mass_props=sim_utils.MassPropertiesCfg(mass=300.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.13, 0.14, 0.145), metallic=0.62, roughness=0.38
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.04)),
    )

    platform_appearance = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/VibrationFloor/LayeredAppearance",
        spawn=PlatformAppearanceCfg(func=spawn_platform_appearance),
    )

    shaker_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ShakerFoundation",
        spawn=ShakerBaseVisualCfg(func=spawn_shaker_base_visuals, geometry=ShakerGeometryCfg()),
    )

    shaker_legs = make_shaker_leg_collection_cfg(ShakerGeometryCfg(), (0.0, 0.0, 0.04), (0.08, -0.13, 0.371))

    robot = _franka_cfg()

    # Collision-enabled camera housing is a child of panda_hand, hence a
    # rigidly mounted part of the hand body rather than a floating view only.
    wrist_camera_model = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        spawn=WristCameraAssemblyCfg(func=spawn_wrist_camera_assembly),
    )

    # A light worktable, separate from the Panda base.  A child UV surface
    # adds photographic marble detail without changing the robust box top.
    worktable = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkTableTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.65, 0.60, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True, max_depenetration_velocity=2.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0010, rest_offset=0.0001),
            mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.19, 0.20), metallic=0.08, roughness=0.52
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.0, 0.34)),
    )

    worktable_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorkTableTop/TexturedSurface",
        spawn=TexturedTableSurfaceCfg(func=spawn_textured_table_surface),
    )

    worktable_appearance = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorkTableTop/LaboratoryFrame",
        spawn=WorktableAppearanceCfg(func=spawn_worktable_appearance),
    )

    table_leg_fl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkTableLegFL",
        spawn=sim_utils.CuboidCfg(
            size=(0.055, 0.055, 0.23),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.22, 0.24), metallic=0.35, roughness=0.40
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.09, -0.245, 0.195)),
    )
    table_leg_fr = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/WorkTableLegFR",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.09, 0.245, 0.195)),
    )
    table_leg_rl = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/WorkTableLegRL",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.245, 0.195)),
    )
    table_leg_rr = table_leg_fl.replace(
        prim_path="{ENV_REGEX_NS}/WorkTableLegRR",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.245, 0.195)),
    )

    workpiece = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            scale=(0.75, 0.75, 0.75),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.5,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0010, rest_offset=0.0001),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, -0.13, 0.47)),
    )

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetBin",
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.16, 0.012),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0008, rest_offset=0.0001),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.86, 0.92), metallic=0.0, roughness=0.48
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.17, 0.376)),
    )

    target_bin_walls = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetBin/Walls",
        spawn=ShallowBinWallsCfg(func=spawn_shallow_bin_walls),
    )

    # Panel-operation task.  All of these stay ``None`` for the default
    # pick-and-place task so its Newton/MJWarp topology is unchanged.
    panel: RigidObjectCfg | None = None
    knob: ArticulationCfg | None = None
    lever: ArticulationCfg | None = None
    button: ArticulationCfg | None = None
    panel_appearance: AssetBaseCfg | None = None

    wrist_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot", update_period=0.0)
    workpiece_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        update_period=0.0,
        track_pose=True,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/WorkTableTop",
            "{ENV_REGEX_NS}/TargetBin",
            "{ENV_REGEX_NS}/Robot/panda_leftfinger",
            "{ENV_REGEX_NS}/Robot/panda_rightfinger",
        ],
    )
    left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Workpiece"],
    )
    right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Workpiece"],
    )
    left_finger_descent_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Workpiece", "{ENV_REGEX_NS}/WorkTableTop"],
    )
    right_finger_descent_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Workpiece", "{ENV_REGEX_NS}/WorkTableTop"],
    )

    knob_contact_left: ContactSensorCfg | None = None
    knob_contact_right: ContactSensorCfg | None = None
    lever_contact_left: ContactSensorCfg | None = None
    lever_contact_right: ContactSensorCfg | None = None
    button_contact_left: ContactSensorCfg | None = None
    button_contact_right: ContactSensorCfg | None = None


def _control_contact_sensor(prim_path: str, finger_paths: list[str]) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=prim_path,
        update_period=0.0,
        track_pose=True,
        filter_prim_paths_expr=finger_paths,
    )


def _configure_panel_task(
    scene: BenchmarkSceneCfg,
    cfg: BenchmarkConfig,
    layout,
    table_top_z: float,
    mu: sim_utils.RigidBodyMaterialCfg,
) -> None:
    """Mount the fixed control panel and its three controls on the current table."""

    del table_top_z, mu  # The console hull is authored by the custom spawner.
    panel_cfg = cfg.panel

    # The close panel-front reach would otherwise be blocked by the physical
    # wrist-camera housing.  Keep it rendered, but disable its collision for
    # this task only; the pick-and-place scene retains the original collider.
    scene.wrist_camera_model = scene.wrist_camera_model.replace(
        spawn=scene.wrist_camera_model.spawn.replace(collision_enabled=False)
    )
    # Newton's joint-wrench sensor currently interprets global body indices as
    # robot-local indices when more than one articulation is present.  The
    # three physical controls therefore make that sensor fail at startup.
    # Panel scoring uses the six link-filtered finger contact sensors instead;
    # pick_place retains its original wrist sensor unchanged.
    scene.wrist_wrench = None

    scene.panel = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ControlPanel",
        spawn=PanelConsoleCollisionCfg(
            func=spawn_panel_console_collision,
            console_depth_m=panel_cfg.console_depth_m,
            console_width_m=panel_cfg.console_width_m,
            console_height_m=panel_cfg.console_height_m,
            front_height_m=panel_cfg.console_front_height_m,
            rear_flat_depth_m=panel_cfg.console_rear_flat_depth_m,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=layout.board_center),
    )
    scene.panel_appearance = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ControlPanel/Appearance",
        spawn=ControlPanelAppearanceCfg(
            func=spawn_control_panel_appearance,
            board_size=panel_cfg.board_size,
            knob_uv=panel_cfg.knob_uv,
            lever_uv=panel_cfg.lever_uv,
            button_uv=panel_cfg.button_uv,
            console_depth_m=panel_cfg.console_depth_m,
            console_width_m=panel_cfg.console_width_m,
            console_height_m=panel_cfg.console_height_m,
            front_height_m=panel_cfg.console_front_height_m,
            rear_flat_depth_m=panel_cfg.console_rear_flat_depth_m,
        ),
    )

    depth = panel_cfg.console_depth_m
    height = panel_cfg.console_height_m
    front_top_z = -0.5 * height + panel_cfg.console_front_height_m
    shoulder_x = 0.5 * depth - panel_cfg.console_rear_flat_depth_m
    slope_dx = shoulder_x + 0.5 * depth
    slope_dz = 0.5 * height - front_top_z
    slope_length = (slope_dx * slope_dx + slope_dz * slope_dz) ** 0.5
    slope_tangent = (slope_dx / slope_length, 0.0, slope_dz / slope_length)
    surface_normal = (-slope_tangent[2], 0.0, slope_tangent[0])

    knob_cx, knob_cy, knob_cz = layout.knob_pivot
    scene.knob = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ControlKnob",
        spawn=PanelControlArticulationCfg(
            func=spawn_panel_control_articulation,
            kind="knob",
            slope_tangent=slope_tangent,
            surface_normal=surface_normal,
            goal=panel_cfg.knob_goal_rad,
            # Review-stage state carrier.  The readable compound geometry is
            # approved render geometry now belongs to the moving link.
            radius_m=panel_cfg.knob_radius_m,
            length_m=panel_cfg.knob_length_m,
            mass_kg=0.06,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(knob_cx, knob_cy, knob_cz),
            joint_pos={"knob_joint": 0.0},
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=["knob_joint"],
                effort_limit_sim=2.0,
                velocity_limit_sim=5.0,
                stiffness=0.0,
                damping=0.08,
            )
        },
    )

    lever_cx, lever_cy, lever_cz = layout.lever_pivot
    scene.lever = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ControlLever",
        spawn=PanelControlArticulationCfg(
            func=spawn_panel_control_articulation,
            kind="lever",
            slope_tangent=slope_tangent,
            surface_normal=surface_normal,
            goal=panel_cfg.lever_goal_rad,
            radius_m=0.011,
            length_m=panel_cfg.lever_length_m,
            mass_kg=0.04,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(lever_cx, lever_cy, lever_cz),
            joint_pos={"lever_joint": 0.0},
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=["lever_joint"],
                effort_limit_sim=2.0,
                velocity_limit_sim=5.0,
                stiffness=0.0,
                damping=0.05,
            )
        },
    )

    button_cx, button_cy, button_cz = layout.button_pivot
    scene.button = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ControlButton",
        spawn=PanelControlArticulationCfg(
            func=spawn_panel_control_articulation,
            kind="button",
            slope_tangent=slope_tangent,
            surface_normal=surface_normal,
            goal=panel_cfg.button_travel_m,
            radius_m=panel_cfg.button_radius_m,
            length_m=panel_cfg.button_length_m,
            mass_kg=0.04,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(button_cx, button_cy, button_cz),
            joint_pos={"button_joint": 0.0},
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=["button_joint"],
                effort_limit_sim=30.0,
                velocity_limit_sim=0.2,
                stiffness=180.0,
                damping=2.5,
            )
        },
    )

    finger_paths = [
        "{ENV_REGEX_NS}/Robot/panda_leftfinger",
        "{ENV_REGEX_NS}/Robot/panda_rightfinger",
    ]
    scene.knob_contact_left = _control_contact_sensor(
        "{ENV_REGEX_NS}/ControlKnob/knob_link", ["{ENV_REGEX_NS}/Robot/panda_leftfinger"]
    )
    scene.knob_contact_right = _control_contact_sensor(
        "{ENV_REGEX_NS}/ControlKnob/knob_link", ["{ENV_REGEX_NS}/Robot/panda_rightfinger"]
    )
    scene.lever_contact_left = _control_contact_sensor(
        "{ENV_REGEX_NS}/ControlLever/lever_link", ["{ENV_REGEX_NS}/Robot/panda_leftfinger"]
    )
    scene.lever_contact_right = _control_contact_sensor(
        "{ENV_REGEX_NS}/ControlLever/lever_link", ["{ENV_REGEX_NS}/Robot/panda_rightfinger"]
    )
    scene.button_contact_left = _control_contact_sensor(
        "{ENV_REGEX_NS}/ControlButton/button_link", ["{ENV_REGEX_NS}/Robot/panda_leftfinger"]
    )
    scene.button_contact_right = _control_contact_sensor(
        "{ENV_REGEX_NS}/ControlButton/button_link", ["{ENV_REGEX_NS}/Robot/panda_rightfinger"]
    )

    # The pick-and-place workpieces, target bin, and their finger-contact
    # sensors are not part of the panel-operation topology.
    scene.workpiece = None
    scene.target = None
    scene.target_bin_walls = None
    scene.workpiece_contact = None
    scene.left_finger_contact = None
    scene.right_finger_contact = None
    scene.left_finger_descent_contact = None
    scene.right_finger_descent_contact = None


CLITE_DRIVER_LABELS = (
    "clite_driver_platform",
    "clite_driver_worktable",
)


def _clite_dynamic_asset(asset_cfg: RigidObjectCfg) -> RigidObjectCfg:
    """Return the asset config with its root body made dynamic for C-lite."""

    rigid_props = asset_cfg.spawn.rigid_props
    if rigid_props is None:
        raise ValueError("C-lite support requires an authored rigid body on every support asset")
    return asset_cfg.replace(
        spawn=asset_cfg.spawn.replace(
            rigid_props=rigid_props.replace(kinematic_enabled=False)
        )
    )


def install_clite_model_constraints(cfg: BenchmarkConfig):
    """Register a MODEL_INIT callback that turns C2 supports into C-lite.

    The callback runs after Newton has built its :class:`ModelBuilder` from the
    USD stage but before model finalization.  It converts the kinematic
    platform/worktable/legs/target bodies to dynamic bodies and welds them to
    fixed-root mocap driver links.  The Panda root stays dynamic and is welded
    to its own mocap driver.
    """

    def _on_model_init(_payload) -> None:
        builder = NewtonManager._builder
        if builder is None:
            raise RuntimeError("Newton model builder was not available at MODEL_INIT")
        if any(str(label).startswith("clite_driver_") for label in builder.body_label):
            return  # already installed (re-entrant callback)

        labels = [str(label) for label in builder.body_label]

        def body_index(suffix: str) -> int:
            matches = [i for i, label in enumerate(labels) if label.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"C-lite expected exactly one body ending in {suffix!r}; "
                    f"got {matches} from {labels}"
                )
            return matches[0]

        platform_idx = body_index("/VibrationFloor")
        worktable_idx = body_index("/WorkTableTop")

        # Main contact supports become dynamic; their free joints remain and
        # the weld equality constraints below drive them through the solver.
        # Table legs and the target bin stay kinematic and keep their original
        # teleport writes to avoid coupled soft-constraint instabilities.
        for index in (platform_idx, worktable_idx):
            builder.body_flags[index] = int(BodyFlags.DYNAMIC)

        def add_driver(label: str, position: tuple[float, float, float]) -> tuple[int, wp.transform]:
            pose = wp.transform((float(position[0]), float(position[1]), float(position[2])), wp.quat_identity())
            driver = builder.add_link(xform=pose, mass=1.0, label=label)
            joint = builder.add_joint_fixed(parent=-1, child=driver)
            builder.joint_X_p[joint] = pose
            builder.add_articulation([joint])
            return driver, pose

        def weld_driver(
            driver: int,
            driver_pose: wp.transform,
            support_idx: int,
            support_position: tuple[float, float, float],
        ) -> None:
            support_pose = wp.transform(
                (float(support_position[0]), float(support_position[1]), float(support_position[2])),
                wp.quat_identity(),
            )
            relpose = wp.transform_inverse(driver_pose) * support_pose
            builder.add_equality_constraint_weld(
                body1=driver,
                body2=support_idx,
                relpose=relpose,
                label=f"clite_weld_{labels[support_idx].rsplit('/', 1)[-1]}",
            )

        platform_driver, platform_pose = add_driver(CLITE_DRIVER_LABELS[0], cfg.platform_center)
        weld_driver(platform_driver, platform_pose, platform_idx, cfg.platform_center)

        worktable_driver, worktable_pose = add_driver(CLITE_DRIVER_LABELS[1], cfg.resolved_worktable_center)
        weld_driver(worktable_driver, worktable_pose, worktable_idx, cfg.resolved_worktable_center)

    NewtonManager.register_callback(
        _on_model_init,
        PhysicsEvent.MODEL_INIT,
        order=-50,
        name="shakebench_clite_model_init",
    )


def make_scene_cfg(cfg: BenchmarkConfig) -> BenchmarkSceneCfg:
    """Instantiate a scene config with selected standard assets and geometry."""

    scene = BenchmarkSceneCfg(
        num_envs=cfg.num_envs,
        env_spacing=1.8,
        replicate_physics=True,
        clone_in_fabric=True,
        lazy_sensor_update=False,
    )
    scene.ground = scene.ground.replace(
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, scene.room.spawn.floor_z_m - scene.room.spawn.pit_depth_m - 0.04)
        )
    )
    mu = sim_utils.RigidBodyMaterialCfg(
        static_friction=cfg.material_mu,
        dynamic_friction=cfg.material_mu,
        restitution=0.0,
    )
    scene.platform = scene.platform.replace(
        spawn=scene.platform.spawn.replace(
            size=cfg.platform_size,
            physics_material=mu,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cfg.contact_margin_m,
                rest_offset=cfg.contact_margin_m,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=cfg.platform_center),
    )
    scene.platform_appearance = scene.platform_appearance.replace(
        spawn=scene.platform_appearance.spawn.replace(
            size_xy=cfg.platform_size[:2],
            top_z_m=0.5 * cfg.platform_size[2] + 0.003,
            robot_xy=cfg.robot_base[:2],
            target_xy=cfg.target_center[:2],
        )
    )
    scene.shaker_base = scene.shaker_base.replace(
        spawn=scene.shaker_base.spawn.replace(geometry=cfg.shaker)
    )
    panel_layout = control_panel_layout(cfg) if cfg.task == "panel_operation" else None
    table_top_z = panel_table_top_z_m(cfg)
    shadow_anchor = (
        (panel_layout.board_center[0], panel_layout.board_center[1], table_top_z + 0.001)
        if panel_layout is not None
        else (cfg.workpiece_start[0], cfg.workpiece_start[1], table_top_z + 0.001)
    )
    scene.shaker_legs = make_shaker_leg_collection_cfg(
        cfg.shaker,
        cfg.platform_center,
        shadow_anchor,
    )
    scene.robot.init_state.pos = cfg.resolved_robot_base
    scene.worktable = scene.worktable.replace(
        spawn=scene.worktable.spawn.replace(
            size=cfg.worktable_size,
            physics_material=mu,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cfg.contact_margin_m,
                rest_offset=cfg.contact_margin_m,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=cfg.resolved_worktable_center),
    )
    scene.worktable_surface = scene.worktable_surface.replace(
        spawn=scene.worktable_surface.spawn.replace(
            size_xy=cfg.worktable_size[:2],
            top_z_m=0.5 * cfg.worktable_size[2] + 0.0004,
        )
    )
    floor_top = cfg.platform_center[2] + 0.5 * cfg.platform_size[2]
    table_bottom = cfg.worktable_center[2] - 0.5 * cfg.worktable_size[2]
    leg_height = table_bottom - floor_top
    leg_z = 0.5 * (table_bottom + floor_top) + cfg.assembly_clearance_m
    leg_x = 0.5 * cfg.worktable_size[0] - 0.055
    leg_y = 0.5 * cfg.worktable_size[1] - 0.055
    leg_xy_local = (
        (-leg_x, -leg_y),
        (-leg_x, leg_y),
        (leg_x, -leg_y),
        (leg_x, leg_y),
    )
    scene.worktable_appearance = scene.worktable_appearance.replace(
        spawn=scene.worktable_appearance.spawn.replace(
            top_size=cfg.worktable_size,
            leg_height_m=leg_height,
            leg_center_z_m=leg_z - cfg.resolved_worktable_center[2],
            leg_xy=leg_xy_local,
        )
    )
    for name, x_sign, y_sign in (
        ("table_leg_fl", -1.0, -1.0),
        ("table_leg_fr", -1.0, 1.0),
        ("table_leg_rl", 1.0, -1.0),
        ("table_leg_rr", 1.0, 1.0),
    ):
        leg = getattr(scene, name)
        position = (
            cfg.resolved_worktable_center[0] + x_sign * leg_x,
            cfg.resolved_worktable_center[1] + y_sign * leg_y,
            leg_z,
        )
        setattr(
            scene,
            name,
            leg.replace(
                spawn=leg.spawn.replace(size=(0.055, 0.055, leg_height)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            ),
        )
    if cfg.task == "panel_operation":
        _configure_panel_task(scene, cfg, panel_layout, table_top_z, mu)
        return scene
    filename = YCB_ASSETS[cfg.assets.workpiece]
    scale = (cfg.assets.workpiece_scale,) * 3
    scene.workpiece = scene.workpiece.replace(
        spawn=scene.workpiece.spawn.replace(
            usd_path=f"{ASSET_ROOT}/Props/YCB/Axis_Aligned_Physics/{filename}",
            scale=scale,
            physics_material=mu,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cfg.contact_margin_m,
                rest_offset=cfg.contact_margin_m,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=cfg.resolved_workpiece_start),
    )
    scene.target = scene.target.replace(
        spawn=scene.target.spawn.replace(
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cfg.contact_margin_m,
                rest_offset=cfg.contact_margin_m,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=cfg.resolved_target_center),
    )
    if cfg.use_clite_support:
        scene.platform = _clite_dynamic_asset(scene.platform)
        scene.worktable = _clite_dynamic_asset(scene.worktable)
    return scene
