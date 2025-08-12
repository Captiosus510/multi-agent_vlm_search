import cv2
import numpy as np
from shapely.geometry import Polygon
import triangle
from scipy.spatial.transform import Rotation as R


def generate_mesh(mask, depth_array, fx, fy, cx, cy, pib_pos):
    # best_mask: binary mask from SAM+CLIP
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Usually there is only 1 large contour, but you can pick the biggest
    floor_contour = max(contours, key=cv2.contourArea)

    mask_world_points = []
    for u, v in floor_contour[:, 0, :]:
        depth = depth_array[v, u]
        if np.isnan(depth) or depth <= 0 or depth > 10:
            depth = 10.0
        # Camera coordinates (same as before)
        X_cam = depth
        Y_cam = -(u - cx) * depth / fx
        Z_cam = -(v - cy) * depth / fy
        P_camera = np.array([X_cam, Y_cam, Z_cam])
        # Rotation and translation (same as before)
        axis = np.array([pib_pos.x_axis, pib_pos.y_axis, pib_pos.z_axis])
        angle = pib_pos.rotation
        r = R.from_rotvec(axis * angle)
        R_matrix = r.as_matrix()
        T = np.array([pib_pos.x, pib_pos.y, pib_pos.z])
        # World coordinates
        P_world = R_matrix @ P_camera + T
        mask_world_points.append(P_world)
    mask_world_points = np.array(mask_world_points)
    
    polygon = Polygon(mask_world_points[:, :2])  # Use only X and Y for 2D polygon

    if not polygon.is_valid:
        print("Polygon is invalid, attempting to fix...")
        polygon = polygon.buffer(0)  # Attempt to fix invalid polygon by buffering with 0 distance
    
    polygon = polygon.simplify(tolerance=0.06, preserve_topology=True)  # Simplify polygon to reduce complexity

    # Get exterior points
    coords = np.array(polygon.exterior.coords) # type: ignore

    # Input to triangle: vertices and segments (edges)
    A = {
        'vertices': coords,
        'segments': [[i, (i + 1) % len(coords)] for i in range(len(coords))],
    }

    # Target max area per triangle (adjust this to control triangle size)
    max_area = 1.2  # smaller = more triangles

    # Triangulate with 'p' (constrained), 'q' (quality), 'a' (area)
    t = triangle.triangulate(A, f'pqa{max_area}')

    # prune the small triangles and number them
    # Compute triangle areas in world XY and remove very small ones, then number remaining

    verts = t['vertices']            # shape (Nv, 2) in world XY
    tris = t['triangles']            # shape (Nt, 3) indices into verts

    # Triangle area via 2D cross product
    areas = []
    for i0, i1, i2 in tris:
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        areas.append(area)
    areas = np.asarray(areas)

    # Keep triangles with area above a small fraction of the target max_area used in triangulation
    min_area_world = 0.15 * max_area   # adjust as needed (5% of target area)
    keep_mask = areas >= min_area_world
    triangles_kept = tris[keep_mask]
    triangle_ids = np.arange(len(triangles_kept))

    # Centroids in world XY for labeling
    centroids_world_xy = verts[triangles_kept].mean(axis=1)  # shape (Nk, 2)

    return triangles_kept, triangle_ids, verts, centroids_world_xy, polygon, mask_world_points

def inverse_mesh(mask_world_points, verts, triangles, triangle_ids, fx, fy, cx, cy, pib_pos):
    # Fit a ground plane z = a*x + b*y + c from the mask_world_points (3D in world frame)
    Xw = mask_world_points[:, 0]
    Yw = mask_world_points[:, 1]
    Zw = mask_world_points[:, 2]
    A = np.c_[Xw, Yw, np.ones_like(Xw)]
    (a, b, c), *_ = np.linalg.lstsq(A, Zw, rcond=None)

    # Lift 2D mesh vertices (world XY) onto the plane to get 3D world points
    mesh_world_xy = verts                     # shape (N, 2)
    mesh_world_z = a * mesh_world_xy[:, 0] + b * mesh_world_xy[:, 1] + c
    mesh_world_3d = np.column_stack([mesh_world_xy, mesh_world_z])

    # Compute inverse extrinsics: world -> camera
    axis = np.array([pib_pos.x_axis, pib_pos.y_axis, pib_pos.z_axis])
    angle = pib_pos.rotation
    R_cw = R.from_rotvec(axis * angle).as_matrix()     # camera -> world
    R_wc = R_cw.T                                      # world -> camera
    T = np.array([pib_pos.x, pib_pos.y, pib_pos.z])    # world translation of camera origin

    # Transform mesh vertices to camera frame
    mesh_cam = (R_wc @ (mesh_world_3d - T).T).T        # shape (N, 3)
    Xc = mesh_cam[:, 0]
    Yc = mesh_cam[:, 1]
    Zc = mesh_cam[:, 2]

    # Project to pixels (your convention uses X as forward axis)
    u = cx - fx * (Yc / Xc)
    v = cy - fy * (Zc / Xc)

    # Keep triangles fully in front of the camera and with finite pixels
    valid_vertex = (Xc > 1e-6) & np.isfinite(u) & np.isfinite(v)

    tri_mask = np.all(valid_vertex[triangles], axis=1)
    triangles_proj = triangles[tri_mask]
    ids_proj = triangle_ids[tri_mask]
    
    # Image-space centroids for each kept triangle (NaN if any vertex invalid)
    pts_img = np.stack([u, v], axis=1)
    valid_tri = np.all(valid_vertex[triangles_proj], axis=1)
    centroids_img = np.full((len(triangles_proj), 2), np.nan, dtype=float)
    centroids_img[valid_tri] = pts_img[triangles_proj[valid_tri]].mean(axis=1)

    return ids_proj, triangles_proj, centroids_img