"""Unit tests for NAO motion library loading and kinematics parity.

Verifies:
  1. MotionLibrary correctly loads data/processed/nao_lafan_subset_50hz.npz in compact format.
  2. Dynamic joint dimension deduction: nj = 24.
  3. Pelvis coupling constraint in reference: q[:, 2] == q[:, 8] (LHipYawPitch == RHipYawPitch).
  4. Frame sampling dictionary matches expected keys and shapes.
  5. Searchsorted clip and bin mappings work as expected.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from allmimic.motion_lib import MotionLibrary
from allmimic.task import NJ

LIB_PATH = "data/processed/nao_lafan_subset_50hz.npz"


@pytest.fixture(scope="module")
def motion_lib():
    assert Path(LIB_PATH).exists(), f"Motion library file not found at {LIB_PATH}"
    return MotionLibrary(LIB_PATH, device="cpu")


def test_library_metadata(motion_lib):
    assert motion_lib.num_clips == 12, f"Expected 12 clips, got {motion_lib.num_clips}"
    assert motion_lib.num_frames == 6000, f"Expected 6000 frames, got {motion_lib.num_frames}"
    assert motion_lib.fps == 50.0
    assert motion_lib.compact is True
    assert motion_lib.q.shape[1] == NJ, f"Expected {NJ} joints, got {motion_lib.q.shape[1]}"
    assert motion_lib.qd.shape[1] == NJ, f"Expected {NJ} joint velocities, got {motion_lib.qd.shape[1]}"
    assert motion_lib.root_pos.shape[1] == 3
    assert motion_lib.root_quat.shape[1] == 4


def test_hip_yaw_pitch_coupling(motion_lib):
    # LHipYawPitch is joint index 2, RHipYawPitch is joint index 8
    l_hyp = motion_lib.q[:, 2]
    r_hyp = motion_lib.q[:, 8]
    diff = torch.abs(l_hyp - r_hyp).max().item()
    # In float16 compact storage, tolerance is float16 epsilon ~ 1e-3
    assert diff < 2e-3, f"Max hip yaw pitch discrepancy in reference exceeds tolerance: {diff}"


def test_frame_sampling(motion_lib):
    g = torch.Generator().manual_seed(42)
    idx = torch.randint(0, motion_lib.num_frames, (64, 32), generator=g)
    frame = motion_lib.frame(idx)

    assert frame["pos"].shape == (64, 32, 3)
    assert frame["rot"].shape == (64, 32, 3, 3)
    assert frame["quat"].shape == (64, 32, 4)
    assert frame["q"].shape == (64, 32, NJ)
    assert frame["v"].shape == (64, 32, 3)
    assert frame["w"].shape == (64, 32, 3)
    assert frame["qd"].shape == (64, 32, NJ)


def test_clip_and_bin_mapping(motion_lib):
    idx = torch.tensor([0, 499, 500, 999, 1000, 5999])
    clips = motion_lib.clip_of(idx)
    expected_clips = torch.tensor([0, 0, 1, 1, 2, 11])
    assert torch.equal(clips, expected_clips), f"Clips mismatch: got {clips}, expected {expected_clips}"

    bins = motion_lib.bin_of(idx)
    assert bins.shape == idx.shape


def test_clamp_and_remaining(motion_lib):
    idx = torch.tensor([490, 498, 499, 500])
    ref = torch.tensor([0, 0, 0, 500])
    rem = motion_lib.remaining(idx)
    assert rem[2] == 0  # 499 is last frame of clip 0
    assert rem[3] == 499  # 500 is first frame of clip 1 (len 500 -> 499 frames remaining)
