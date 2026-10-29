#!/usr/bin/env python3
"""Build authentic binary CAD assets for the NAO robot directly from MuJoCo's MjModel.

Extracts:
    - Pre-centered, scaled vertices (m.mesh_vert)
    - Vertex normals (m.mesh_normal)
    - Texture coordinates / UVs (m.mesh_texcoord)
    - Face indices (m.mesh_face)

Generates:
    - web_dep/public/nao_cad.json: Visual geoms metadata and buffer offsets.
    - web_dep/public/nao_cad.bin: Packed binary vertex (pos+normal+uv) and index buffer.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

XML_PATH = ROOT / "assets" / "nao" / "nao_visual_scene.xml"
OUT_DIR = ROOT / "web_dep" / "public"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_file = OUT_DIR / "nao_cad.bin"
    json_file = OUT_DIR / "nao_cad.json"

    print(f"[*] Loading MuJoCo model from {XML_PATH}")
    m = mujoco.MjModel.from_xml_path(str(XML_PATH))

    # All visual geoms (group 2), excluding individual clutter finger meshes
    vis_geom_ids = [
        g for g in range(m.ngeom)
        if m.geom_group[g] == 2
        and not ("Finger" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
                 or "Thumb" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]))
    ]
    print(f"[*] Selected {len(vis_geom_ids)} main visual geoms directly from MjModel.")

    packed_vertex_bytes = bytearray()
    packed_index_bytes = bytearray()

    vert_offset_bytes = 0
    index_offset_bytes = 0

    geoms_meta = []
    mesh_cache = {}

    for g in vis_geom_ids:
        mid = int(m.geom_dataid[g])
        mname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        bid = int(m.geom_bodyid[g])
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)

        if mid not in mesh_cache:
            v_adr = m.mesh_vertadr[mid]
            v_num = m.mesh_vertnum[mid]
            f_adr = m.mesh_faceadr[mid]
            f_num = m.mesh_facenum[mid]

            verts = m.mesh_vert[v_adr : v_adr + v_num].astype(np.float32)
            normals = m.mesh_normal[v_adr : v_adr + v_num].astype(np.float32)
            faces = m.mesh_face[f_adr : f_adr + f_num].astype(np.uint32)

            # Texture coords (UVs)
            if m.mesh_texcoordadr[mid] >= 0:
                t_adr = m.mesh_texcoordadr[mid]
                uvs = m.mesh_texcoord[t_adr : t_adr + v_num].astype(np.float32)
            else:
                uvs = np.zeros((v_num, 2), dtype=np.float32)

            # Interleaved vertex layout: [x, y, z, nx, ny, nz, u, v] (8 floats = 32 bytes)
            interleaved = np.hstack([verts, normals, uvs]).astype(np.float32)

            v_start = vert_offset_bytes
            v_bytes = interleaved.tobytes()
            packed_vertex_bytes.extend(v_bytes)
            vert_offset_bytes += len(v_bytes)

            f_start = index_offset_bytes
            f_bytes = faces.tobytes()
            packed_index_bytes.extend(f_bytes)
            index_offset_bytes += len(f_bytes)

            mesh_cache[mid] = {
                "vert_byte_offset": v_start,
                "vert_count": int(v_num),
                "index_byte_offset": f_start,
                "index_count": int(f_num * 3),
            }

        meta = mesh_cache[mid]
        geoms_meta.append({
            "geom_id": int(g),
            "body_id": bid,
            "body_name": bname,
            "mesh_name": mname,
            "vert_byte_offset": meta["vert_byte_offset"],
            "vert_count": meta["vert_count"],
            "index_byte_offset": meta["index_byte_offset"],
            "index_count": meta["index_count"],
        })

    vert_total_len = len(packed_vertex_bytes)
    with open(bin_file, "wb") as f:
        f.write(struct.pack("<Q", vert_total_len))
        f.write(packed_vertex_bytes)
        f.write(packed_index_bytes)

    total_mb = bin_file.stat().st_size / (1024 * 1024)
    print(f"[+] Direct MjModel CAD Binary written to {bin_file} ({total_mb:.2f} MB)")

    out_json = {
        "model": "SoftBank NAO V40 Humanoid",
        "total_geoms": len(geoms_meta),
        "stride_floats": 8,  # pos(3) + norm(3) + uv(2)
        "stride_bytes": 32,
        "vert_buffer_bytes": vert_total_len,
        "geoms": geoms_meta
    }

    with open(json_file, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[+] CAD JSON metadata written to {json_file}")


if __name__ == "__main__":
    main()
