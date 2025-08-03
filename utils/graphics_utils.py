#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
import open3d as o3d

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

# from https://github.com/threestudio-project/threestudio
class Mesh:
    def __init__(
        self, v_pos, t_pos_idx, **kwargs
    ) -> None:
        self.v_pos = v_pos
        self.t_pos_idx = t_pos_idx
        self._v_nrm = None
        self._v_tng = None
        self._v_tex = None
        self._t_tex_idx = None
        self._v_rgb = None
        self._edges = None
        self.extras = {}
        for k, v in kwargs.items():
            self.add_extra(k, v)

    def add_extra(self, k, v) -> None:
        self.extras[k] = v

    def remove_outlier(self, outlier_n_faces_threshold):

        # use trimesh to first split the mesh into connected components
        # then remove the components with less than n_face_threshold faces
        import trimesh

        # construct a trimesh object
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )

        # split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # log the number of faces in each component

        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(
                max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold
            )
        else:
            # set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold

        # remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]

        # merge the components
        mesh = trimesh.util.concatenate(components)

        # convert back to our mesh format
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)

        clean_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        if len(self.extras) > 0:
            clean_mesh.extras = self.extras

        return clean_mesh
    
     
def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = (
        covs[:, 0],
        covs[:, 1],
        covs[:, 2],
        covs[:, 3],
        covs[:, 4],
        covs[:, 5],
    )

    # eps must be small enough !!!
    inv_det = 1 / (
        a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24
    )
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = (
        -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f)
        - x * y * inv_b
        - x * z * inv_c
        - y * z * inv_e
    )

    power[power > 0] = -1e10  # abnormal values... make weights 0

    return torch.exp(power)


def decimate_mesh(
    verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        import pymeshlab as pml

        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.PercentageValue(1)
            )

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(
        f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
):
    # verts: [N, 3]
    # faces: [N, 3]
    import pymeshlab as pml

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.PercentageValue(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.PercentageValue(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.PureValue(remesh_size)
        )

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces

def process_mesh_to_o3d(mesh):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.v_pos.detach().cpu().numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.t_pos_idx.detach().cpu().numpy())
    return mesh_o3d