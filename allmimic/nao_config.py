"""NAO robot configuration and mjlab EntityCfg for SuperMimic."""

from pathlib import Path
import re
import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg

NAO_SUPERMIMIC_XML = Path(__file__).resolve().parent.parent / "assets" / "nao" / "nao_supermimic.xml"

# Per-subsystem action scales
NAO_ACTION_SCALE = {
    r'.*HipRoll.*': 0.30,
    r'.*AnkleRoll.*': 0.28,
    r'.*Hip.*': 0.25,
    r'.*Knee.*': 0.25,
    r'.*Ankle.*': 0.25,
    r'.*Shoulder.*': 0.35,
    r'.*Elbow.*': 0.35,
    r'.*Wrist.*': 0.35,
    r'Head.*': 0.35,
}

NAO_ROOT_BODY = "torso"
NAO_FOOT_GEOM_RE = r"^(l_foot|r_foot)$"

# Initial standing pose with slightly bent knees for stable balance
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.334),
    joint_pos={
        "HeadPitch": 0.0,
        "HeadYaw": 0.0,
        ".*ShoulderPitch": 1.3963,
        "LShoulderRoll": 0.3491,
        "RShoulderRoll": -0.3491,
        "LElbowYaw": -1.3963,
        "LElbowRoll": -1.0472,
        "RElbowYaw": 1.3963,
        "RElbowRoll": 1.0472,
        ".*WristYaw": 0.0,
        ".*HipYawPitch": 0.0,
        ".*HipRoll": 0.0,
        ".*HipPitch": -0.3491,
        ".*KneePitch": 0.6981,
        ".*AnklePitch": -0.3491,
        ".*AnkleRoll": 0.0,
    },
    joint_vel={".*": 0.0},
)

# Foot contact primitives
_foot_regex = "^(l_foot|r_foot)$"
_torso_regex = "^torso_collision$"
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex, _torso_regex),
    condim={_foot_regex: 3, _torso_regex: 1},
    priority={_foot_regex: 1, _torso_regex: 0},
    friction={_foot_regex: (1.2,)},
    contype=1,
    conaffinity=0,
    disable_other_geoms=True,
)

NAO_ACTUATOR_CFGS = (
    # Leg Actuators
    BuiltinPositionActuatorCfg(
        target_names_expr=(r".*Hip.*",),
        stiffness=20.0,
        damping=5.0,
        effort_limit=8.0,
        armature=0.005,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(r".*Knee.*",),
        stiffness=18.0,
        damping=4.0,
        effort_limit=5.0,
        armature=0.005,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(r".*Ankle.*",),
        stiffness=12.0,
        damping=1.5,
        effort_limit=5.0,
        armature=0.005,
    ),
    # Arm Actuators
    BuiltinPositionActuatorCfg(
        target_names_expr=(r".*Shoulder.*",),
        stiffness=10.0,
        damping=5.0,
        effort_limit=5.0,
        armature=0.005,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(r".*Elbow.*",),
        stiffness=2.0,
        damping=1.0,
        effort_limit=5.0,
        armature=0.005,
    ),
    # Head Actuators
    BuiltinPositionActuatorCfg(
        target_names_expr=(r"Head.*",),
        stiffness=2.0,
        damping=2.0,
        effort_limit=8.0,
        armature=0.005,
    ),
    # Wrist Actuators
    BuiltinPositionActuatorCfg(
        target_names_expr=(r".*Wrist.*",),
        stiffness=2.0,
        damping=0.1,
        effort_limit=1.0,
        armature=0.005,
    ),
)

NAO_ACTUATOR_DELAYED_CFGS = tuple(
    DelayedActuatorCfg(
        base_cfg=cfg,
        delay_min_lag=0,
        delay_max_lag=2,
        delay_update_period=0,
    )
    for cfg in NAO_ACTUATOR_CFGS
)

NAO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=NAO_ACTUATOR_CFGS,
    soft_joint_pos_limit_factor=0.95,
)

NAO_ARTICULATION_DELAYED = EntityArticulationInfoCfg(
    actuators=NAO_ACTUATOR_DELAYED_CFGS,
    soft_joint_pos_limit_factor=0.95,
)


def get_nao_spec() -> mujoco.MjSpec:
    assert NAO_SUPERMIMIC_XML.exists(), f"NAO XML not found at {NAO_SUPERMIMIC_XML}"
    spec = mujoco.MjSpec.from_file(str(NAO_SUPERMIMIC_XML))
    for a in list(spec.actuators):
        spec.delete(a)
    for i, g in enumerate(spec.geoms):
        if not g.name:
            mname = g.meshname if hasattr(g, "meshname") else ""
            g.name = f"{mname}_col" if mname else f"geom_{i}"
    return spec


def get_nao_supermimic_robot_cfg(delayed: bool = False) -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_nao_spec,
        articulation=NAO_ARTICULATION_DELAYED if delayed else NAO_ARTICULATION,
    )
