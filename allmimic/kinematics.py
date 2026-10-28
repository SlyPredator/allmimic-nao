"""Batched, differentiable forward kinematics for a floating-base tree of hinge joints.

Built from a compiled MjModel. Bodies are processed level by level so the cost is
the tree depth, not the number of bodies. Body velocities are the velocities of the
body frame origins in the world frame (same as mjlab's body_link_*_vel_w).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .rotations import axis_angle_to_mat


class TreeKinematics:
    def __init__(self, mj_model: mujoco.MjModel, device: str, root_body: int = 1):
        m = mj_model
        self.device = device
        r = root_body
        assert m.jnt_type[m.body_jntadr[r]] == mujoco.mjtJoint.mjJNT_FREE
        # The robot's bodies are contiguous in MuJoCo's depth-first order.
        bodies = [r]
        for b in range(r + 1, m.nbody):
            if m.body_rootid[b] == r:
                bodies.append(b)
        assert bodies == list(range(r, r + len(bodies)))
        nb = len(bodies)
        j_root = m.body_jntadr[r]
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).split('/')[-1] for b in bodies]
        parent = np.array([0] + [m.body_parentid[b] - r for b in bodies[1:]])
        joint_of_body = np.full(nb, -1)
        axes = np.zeros((nb, 3))
        for b in bodies[1:]:
            assert m.body_jntnum[b] == 1, f'body {b} has {m.body_jntnum[b]} joints'
            j = m.body_jntadr[b]
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            assert np.allclose(m.jnt_pos[j], 0.0), 'hinge must sit at the body origin'
            # Hinge index among the robot's non-free joints, in MuJoCo joint order.
            joint_of_body[b - r] = j - j_root - 1
            axes[b - r] = m.jnt_axis[j]
        self.joint_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).split('/')[-1] for j in range(j_root + 1, j_root + nb)
        ]
        depth = np.zeros(nb, dtype=int)
        for b in range(1, nb):
            depth[b] = depth[parent[b]] + 1
        self.levels = []
        for d in range(1, depth.max() + 1):
            ids = np.nonzero(depth == d)[0]
            self.levels.append(
                (
                    torch.tensor(ids, device=device),
                    torch.tensor(parent[ids], device=device),
                    torch.tensor(joint_of_body[ids], device=device),
                )
            )
        self.nb = nb
        pos = torch.tensor(np.array([m.body_pos[b] for b in bodies]), dtype=torch.float32)
        quat = torch.tensor(np.array([m.body_quat[b] for b in bodies]), dtype=torch.float32)
        from .rotations import quat_to_mat

        self.offset_pos = pos.to(device)
        self.offset_rot = quat_to_mat(quat).to(device)
        self.axes = torch.tensor(axes, dtype=torch.float32, device=device)
        # The root body's own offset is irrelevant: the free joint sets its pose.

        # Site tracking metadata
        self.site_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)]
        self.site_body = [m.site_bodyid[s] - r for s in range(m.nsite)]
        self.site_pos = torch.tensor(np.array([m.site_pos[s] for s in range(m.nsite)]), dtype=torch.float32, device=device)
        self.site_rot = quat_to_mat(torch.tensor(np.array([m.site_quat[s] for s in range(m.nsite)]), dtype=torch.float32)).to(device)

    def body_index(self, names) -> list[int]:
        return [self.body_names.index(n) for n in names]

    def site_index(self, names) -> list[int]:
        return [self.site_names.index(n) for n in names]

    def forward_sites(self, P: torch.Tensor, R: torch.Tensor, site_ids: list[int] | torch.Tensor | None = None):
        """Computes site positions (..., ns, 3) and rotations (..., ns, 3, 3) from body poses P and R."""
        if site_ids is None:
            site_ids = list(range(len(self.site_names)))
        if isinstance(site_ids, torch.Tensor):
            site_ids = site_ids.tolist()
        bids = [self.site_body[s] for s in site_ids]
        spos = self.site_pos[site_ids]
        srot = self.site_rot[site_ids]
        Pb = P.index_select(-2, torch.tensor(bids, device=P.device))
        Rb = R.index_select(-3, torch.tensor(bids, device=R.device))
        Ps = Pb + (Rb @ spos.unsqueeze(-1)).squeeze(-1)
        Rs = Rb @ srot
        return Ps, Rs

    def forward(
        self,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        q: torch.Tensor,
        root_lin_vel: torch.Tensor | None = None,
        root_ang_vel: torch.Tensor | None = None,
        qd: torch.Tensor | None = None,
    ):
        """Returns body positions (..., nb, 3), rotations (..., nb, 3, 3) and, when
        velocities are given, body linear and angular velocities (..., nb, 3)."""
        batch = root_pos.shape[:-1]
        with_vel = root_lin_vel is not None
        # Work with stacked tensors per level to keep the op count at O(depth).
        P = root_pos.new_zeros(batch + (self.nb, 3))
        R = root_pos.new_zeros(batch + (self.nb, 3, 3))
        P = P.index_copy(-2, torch.tensor([0], device=P.device), root_pos.unsqueeze(-2))
        R = R.index_copy(-3, torch.tensor([0], device=P.device), root_rot.unsqueeze(-3))
        if with_vel:
            V = root_pos.new_zeros(batch + (self.nb, 3))
            W = root_pos.new_zeros(batch + (self.nb, 3))
            V = V.index_copy(-2, torch.tensor([0], device=P.device), root_lin_vel.unsqueeze(-2))
            W = W.index_copy(-2, torch.tensor([0], device=P.device), root_ang_vel.unsqueeze(-2))
        for ids, par, jid in self.levels:
            Rp = R.index_select(-3, par)
            Pp = P.index_select(-2, par)
            qa = q.index_select(-1, jid)
            axes = self.axes[ids]
            Rj = axis_angle_to_mat(axes * qa.unsqueeze(-1))
            Rc = Rp @ self.offset_rot[ids] @ Rj
            Pc = Pp + (Rp @ self.offset_pos[ids].unsqueeze(-1)).squeeze(-1)
            P = P.index_copy(-2, ids, Pc)
            R = R.index_copy(-3, ids, Rc)
            if with_vel:
                Vp = V.index_select(-2, par)
                Wp = W.index_select(-2, par)
                qda = qd.index_select(-1, jid)
                axis_w = (Rc @ axes.unsqueeze(-1)).squeeze(-1)
                Wc = Wp + axis_w * qda.unsqueeze(-1)
                Vc = Vp + torch.cross(Wp, Pc - Pp, dim=-1)
                V = V.index_copy(-2, ids, Vc)
                W = W.index_copy(-2, ids, Wc)
        if with_vel:
            return P, R, V, W
        return P, R


class FastTreeKinematics(TreeKinematics):
    """Same outputs as TreeKinematics, unrolled per body. Supports both principal
    axes and arbitrary 3D joint axes (such as NAO's 45-deg HipYawPitch)."""

    def __init__(self, mj_model, device, root_body: int = 1):
        super().__init__(mj_model, device, root_body)
        self.parent_list = [0] * self.nb
        self.joint_list = [0] * self.nb
        for ids, par, jid in self.levels:
            for i, p, j in zip(ids.tolist(), par.tolist(), jid.tolist()):
                self.parent_list[i] = p
                self.joint_list[i] = j
        ax = self.axes.cpu()
        self.axis_k = [-1] * self.nb
        self.axis_sign = [1.0] * self.nb
        for b in range(1, self.nb):
            k = int(ax[b].abs().argmax())
            if torch.allclose(ax[b].abs(), torch.nn.functional.one_hot(torch.tensor(k), 3).float(), atol=1e-4):
                self.axis_k[b] = k
                self.axis_sign[b] = 1.0 if ax[b, k] > 0 else -1.0
            else:
                self.axis_k[b] = -1
        self.O = [self.offset_rot[b] for b in range(self.nb)]
        self.off = [self.offset_pos[b] for b in range(self.nb)]

    @staticmethod
    def _mv(R, x):
        return (R * x.unsqueeze(-2)).sum(-1)

    @staticmethod
    def _mm(A, B):
        return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)

    def forward(self, root_pos, root_rot, q, root_lin_vel=None, root_ang_vel=None, qd=None):
        with_vel = root_lin_vel is not None
        P = [root_pos] + [None] * (self.nb - 1)
        R = [root_rot] + [None] * (self.nb - 1)
        if with_vel:
            V = [root_lin_vel] + [None] * (self.nb - 1)
            W = [root_ang_vel] + [None] * (self.nb - 1)
        for b in range(1, self.nb):
            p, j, k = self.parent_list[b], self.joint_list[b], self.axis_k[b]
            Rp = R[p]
            O = self.O[b]
            if k >= 0:
                sgn = self.axis_sign[b]
                q_val = q[..., j] * sgn if sgn != 1.0 else q[..., j]
                c, s = torch.cos(q_val), torch.sin(q_val)
                # A = O @ Rot_k(q): column k unchanged, the other two rotate.
                i1, i2 = [(1, 2), (2, 0), (0, 1)][k]
                c, s = c[..., None], s[..., None]
                cols = [None, None, None]
                cols[k] = O[:, k].expand(c.shape[:-1] + (3,))
                cols[i1] = c * O[:, i1] + s * O[:, i2]
                cols[i2] = -s * O[:, i1] + c * O[:, i2]
                A = torch.stack(cols, dim=-1)
                Rc = self._mm(Rp, A)
                axis_w = Rc[..., :, k] * sgn
            else:
                axis = self.axes[b]
                Rj = axis_angle_to_mat(axis * q[..., j : j + 1])
                A = self._mm(O, Rj)
                Rc = self._mm(Rp, A)
                axis_w = self._mv(Rc, axis)
            Pc = P[p] + self._mv(Rp, self.off[b])
            P[b], R[b] = Pc, Rc
            if with_vel:
                Wc = W[p] + axis_w * qd[..., j : j + 1]
                Vc = V[p] + torch.cross(W[p], Pc - P[p], dim=-1)
                V[b], W[b] = Vc, Wc
        out = (torch.stack(P, -2), torch.stack(R, -3))
        if with_vel:
            out = out + (torch.stack(V, -2), torch.stack(W, -2))
        return out

