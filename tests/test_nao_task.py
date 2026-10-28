"""Unit and regression tests for NAO Task, Dimensions, and Action Projection."""

import pytest
import torch
import mujoco

from allmimic.kinematics import FastTreeKinematics
from allmimic.networks import ActionProjection, Policy, apply_delta
from allmimic.rotations import quat_to_mat
from allmimic.task import (
    GOAL_FRAME_DIM,
    NJ,
    NU,
    OBS_DIM,
    XI_DIM,
    Task,
    TaskCfg,
    expand_actions,
    flat_to_state,
    state_to_flat,
)
from allmimic.nao_config import NAO_SUPERMIMIC_XML, NAO_ACTION_SCALE


@pytest.fixture(scope="module")
def task_fixture():
    m = mujoco.MjModel.from_xml_path(str(NAO_SUPERMIMIC_XML))
    kin = FastTreeKinematics(m, "cpu")
    q0 = torch.zeros(NJ)
    lim = torch.stack([-torch.ones(NJ) * 1.5, torch.ones(NJ) * 1.5], dim=-1)
    scale = torch.full((NJ,), 0.25)
    cfg = TaskCfg()
    task = Task(kin, q0, lim, scale, cfg, "cpu")
    return task, kin, m


def test_dimensions():
    assert NJ == 24
    assert NU == 23
    assert OBS_DIM == 24 + 24 + 3 + 3 + 23  # 77
    assert GOAL_FRAME_DIM == 3 + 6 + 24 + 24 + 3 + 3  # 63
    assert XI_DIM == 1 + 24 + 3  # 28


def test_flat_state_roundtrip():
    B = 16
    pos = torch.randn(B, 3)
    rot = quat_to_mat(torch.randn(B, 4))
    q = torch.randn(B, NJ)
    v = torch.randn(B, 3)
    w = torch.randn(B, 3)
    qd = torch.randn(B, NJ)

    s = {"pos": pos, "rot": rot, "q": q, "v": v, "w": w, "qd": qd}
    flat = state_to_flat(s)
    assert flat.shape == (B, 61)  # 3 + 4 + 24 + 3 + 3 + 24 = 61

    s_rec = flat_to_state(flat)
    assert torch.allclose(s_rec["pos"], pos)
    assert torch.allclose(s_rec["q"], q)
    assert torch.allclose(s_rec["v"], v)
    assert torch.allclose(s_rec["w"], w)
    assert torch.allclose(s_rec["qd"], qd)


def test_action_expansion_and_projection():
    B = 32
    a = torch.randn(B, NU)
    u = expand_actions(a)
    assert u.shape == (B, NJ)
    # Channel 2 maps to LHipYawPitch (joint 2) and RHipYawPitch (joint 8)
    assert torch.allclose(u[:, 2], a[:, 2])
    assert torch.allclose(u[:, 8], a[:, 2])
    assert torch.allclose(u[:, 3:8], a[:, 3:8])
    assert torch.allclose(u[:, 9:14], a[:, 8:13])
    assert torch.allclose(u[:, 14:24], a[:, 13:23])

    proj = ActionProjection()
    u_proj = proj(a)
    assert torch.allclose(u, u_proj)


def test_world_model_hip_symmetrization():
    B = 16
    s = {
        "pos": torch.zeros(B, 3),
        "rot": torch.eye(3).expand(B, 3, 3),
        "v": torch.zeros(B, 3),
        "w": torch.zeros(B, 3),
        "q": torch.zeros(B, NJ),
        "qd": torch.zeros(B, NJ),
    }
    # Create delta with asymmetric hip changes: joint 2 = +0.4, joint 8 = -0.2
    d = torch.zeros(B, 3 + 3 + 3 + 3 + NJ + NJ)
    d[:, 12 + 2] = 0.4
    d[:, 12 + 8] = -0.2
    d[:, 12 + NJ + 2] = 1.0
    d[:, 12 + NJ + 8] = 0.0

    s1 = apply_delta(s, d)
    # Expected mean: (0.4 + -0.2) / 2 = 0.1
    expected_q_hyp = torch.full((B,), 0.1)
    assert torch.allclose(s1["q"][:, 2], expected_q_hyp)
    assert torch.allclose(s1["q"][:, 8], expected_q_hyp)

    # Expected qd mean: (1.0 + 0.0) / 2 = 0.5
    expected_qd_hyp = torch.full((B,), 0.5)
    assert torch.allclose(s1["qd"][:, 2], expected_qd_hyp)
    assert torch.allclose(s1["qd"][:, 8], expected_qd_hyp)


def test_policy_decoder_dimension():
    pol = Policy(OBS_DIM, GOAL_FRAME_DIM * 10)
    obs = torch.randn(8, OBS_DIM)
    goal = torch.randn(8, GOAL_FRAME_DIM * 10)
    out = pol(obs, goal)
    assert out.shape == (8, NU)  # Exactly 23 policy actions


def test_reward_computation(task_fixture):
    task, kin, m = task_fixture
    B = 4
    s = {
        "pos": torch.zeros(B, 3),
        "rot": torch.eye(3).expand(B, 3, 3),
        "q": torch.zeros(B, NJ),
        "v": torch.zeros(B, 3),
        "w": torch.zeros(B, 3),
        "qd": torch.zeros(B, NJ),
    }
    ref = {k: v.clone() for k, v in s.items()}
    kin_out = kin.forward(s["pos"], s["rot"], s["q"], s["v"], s["w"], s["qd"])
    ref_kin = kin.forward(ref["pos"], ref["rot"], ref["q"], ref["v"], ref["w"], ref["qd"])
    action = torch.zeros(B, NU)
    prev_action = torch.zeros(B, NU)

    r, terms = task.reward(s, kin_out, ref, ref_kin, action, prev_action)
    assert r.shape == (B,)
    assert not torch.isnan(r).any()
    # At exact reference match, reward should be high
    assert (r > 0.0).all()
