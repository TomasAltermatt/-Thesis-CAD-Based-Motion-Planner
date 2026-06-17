import trimesh
import pyvista as pv
import numpy as np
import networkx as nx
import os
import pandas as pd
import time
from pathlib import Path
from classes import PseudoFace
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, Point
from itertools import product, permutations

# ----------------------------------------------------- MAIN FUNCTIONS ----------------------------------------------
## AABB overlap test functions

def check_2d_aabb_overlap(bounds_a, bounds_b, extraction_axis):
    overlap_region = {}
    # Note: This check considers the bounding boxes as inputted here, so the orientation depends
    # on how the bounds are defined before calling the function. 
    # For this check it's based on the oriented bounding boxes of part_a, and part_b is transformed
    # to match that orientation. As such it checks the extraction of part_a along its oriented bounding box 
    # axes. If you want to check the extraction along the world axes, you would need to ensure that the bounds
    # are defined in the world coordinate system before calling this function.
    """
    Squashes 3D bounding boxes onto a 2D plane based on the extraction axis
    and checks if the 2D rectangles overlap.
    extraction_axis: 0 for X, 1 for Y, 2 for Z
    """

    # Figure out which two axes form our 2D "shadow" plane
    # If we extract in Z (2), our 2D plane uses X (0) and Y (1).
    axis_idx = {"x": 0, "y": 1, "z": 2}
    all_axes = [0, 1, 2]
    all_axes.remove(axis_idx[extraction_axis])
    u_axis = all_axes[0]
    v_axis = all_axes[1]
    w_axis = axis_idx[extraction_axis]
    
    # Extract the Min and Max for the U axis (e.g., the X axis)
    # bounds[0] is Min, bounds[1] is Max
    a_min_u, a_max_u = bounds_a[0][u_axis], bounds_a[1][u_axis]
    b_min_u, b_max_u = bounds_b[0][u_axis], bounds_b[1][u_axis]
    
    # Extract the Min and Max for the V axis (e.g., the Y axis)
    a_min_v, a_max_v = bounds_a[0][v_axis], bounds_a[1][v_axis]
    b_min_v, b_max_v = bounds_b[0][v_axis], bounds_b[1][v_axis]
    
    # Calculate the exact boundaries of the overlap region (The SR)
    overlap_min_u = max(a_min_u, b_min_u)
    overlap_max_u = min(a_max_u, b_max_u)

    overlap_min_v = max(a_min_v, b_min_v)
    overlap_max_v = min(a_max_v, b_max_v)

    overlap_region['overlap_u'] = (overlap_min_u, overlap_max_u)
    overlap_region['overlap_v'] = (overlap_min_v, overlap_max_v)

    # Case 1: Overlap does not exist at all (AABBs don't even touch)  --> Return 0
    if not ((overlap_min_u <= overlap_max_u) and (overlap_min_v <= overlap_max_v)):
          # No overlap
        return (overlap_region, 0)
    
    # Check COAABB overlap
    a_lims = [(a_min_u, a_max_u), (a_min_v, a_max_v)]
    b_lims = [(b_min_u, b_max_u), (b_min_v, b_max_v)]
    coaabb_overlap = check_COAABB_overlap(a_lims, b_lims)

    # Case 2: AABBs overlap but COAABBs do not (We need to check PFs) --> Return -2
    if not coaabb_overlap:
        return (overlap_region, -2)

    a_min_w, a_max_w = bounds_a[0][w_axis], bounds_a[1][w_axis]
    b_min_w, b_max_w = bounds_b[0][w_axis], bounds_b[1][w_axis]

    # Case 3: AABBs overlap and COAABBs overlap
    overlap_result = None
    if a_min_w >= b_max_w:
         overlap_result = -1 # Part A cannot be extracted in the negative extraction direction without colliding with B
    elif b_min_w >= a_max_w:
        overlap_result = 1   # Part A cannot be extracted in extraction direction without colliding with B, 

    else:
        overlap_result = 2   # Part A cannot be extracted in either direction without colliding with B

    return (overlap_region, overlap_result)

    # Note: The return values are as follows:
    #  0: No overlap at all (AABBs don't even touch)
    # -2: AABBs overlap but COAABBs do not (We need to check PFs)
    # -1: A cannot be extracted in the negative extraction direction without colliding with B
    #  1: A cannot be extracted in the positive extraction direction without colliding with B, 
    #    but can be extracted in the negative direction
    #  2: A cannot be extracted in either direction without colliding with B

def check_COAABB_overlap(a_lims, b_lims, epsilon = 0.05):
    a_min_u, a_max_u = a_lims[0]
    a_min_v, a_max_v = a_lims[1]
    b_min_u, b_max_u = b_lims[0]
    b_min_v, b_max_v = b_lims[1]

    # Define lu and lv
    lu = min(b_max_u - b_min_u, a_max_u - a_min_u)
    lv = min(b_max_v - b_min_v, a_max_v - a_min_v)

    # Conditions for COAABB overlap
    cond1 = (a_min_u - b_min_u) >= -epsilon*lu and (b_min_v - a_min_v) >= -epsilon*lv
    cond2 = (b_max_u - a_max_u) >= -epsilon*lu and (a_max_v - b_max_v) >= -epsilon*lv
    cond3 = (b_min_u - a_min_u) >= -epsilon*lu and (a_min_v - b_min_v) >= -epsilon*lv
    cond4 = (a_max_u - b_max_u) >= -epsilon*lu and (b_max_v - a_max_v) >= -epsilon*lv

    return (cond1 and cond2) or (cond3 and cond4)


## Pseudo Face overlap test functions
def create_PFs(part: trimesh.Trimesh, extraction_axis: str, tolerance = 1e-4):
    axis_idx = {"x": 0, "y": 1, "z": 2}
    all_axes = [0, 1, 2]
    all_axes.remove(axis_idx[extraction_axis])
    u_axis = all_axes[0]
    v_axis = all_axes[1]

    # Get face normals for the part in line with the extraction axis
    normals_w = part.face_normals[:, axis_idx[extraction_axis]]
    valid_faces_mask = np.abs(normals_w) > tolerance
    valid_face_indices = np.where(valid_faces_mask)[0]
    
    # Check if the left and right triangles of face adjacency are valid
    left_valid_mask = np.isin(part.face_adjacency[:, 0], valid_face_indices)
    right_valid_mask = np.isin(part.face_adjacency[:, 1], valid_face_indices)
    both_valid_mask = left_valid_mask & right_valid_mask
    valid_pairs = part.face_adjacency[both_valid_mask]

    # Create graph to find connected triangles via valid_pairs
    G = nx.Graph()
    G.add_edges_from(valid_pairs)

    return [PseudoFace(part, c, extraction_axis) for c in list(nx.connected_components(G))]

def check_PF_overlap(pf_a: PseudoFace, pf_b: PseudoFace):
    result = [-2, -2]
    # Dynamic axis selection using the class attribute
    w_idx = pf_a.extraction_axis

    # Pseudoface A bounding box 2D limits and dynamic depth limits
    a_min_u, a_min_v = pf_a.triangles_2d.min(axis=(0,1))
    a_max_u, a_max_v = pf_a.triangles_2d.max(axis=(0,1))
    
    # Squash axis 0 (triangles) and axis 1 (vertices) to get true 3D bounding box
    a_min_w = pf_a.triangles_3d[:, :, w_idx].min()
    a_max_w = pf_a.triangles_3d[:, :, w_idx].max()

    # Pseudoface B bounding box 2D limits and dynamic depth limits
    b_min_u, b_min_v = pf_b.triangles_2d.min(axis=(0,1))
    b_max_u, b_max_v = pf_b.triangles_2d.max(axis=(0,1))
    
    b_min_w = pf_b.triangles_3d[:, :, w_idx].min()
    b_max_w = pf_b.triangles_3d[:, :, w_idx].max()

    # Calculate overlap region in 2D
    overlap_min_u = max(a_min_u, b_min_u)
    overlap_max_u = min(a_max_u, b_max_u)
    overlap_min_v = max(a_min_v, b_min_v)
    overlap_max_v = min(a_max_v, b_max_v)

    # Filter 2: Check if the 2D bounding boxes of the pseudo-faces overlap
    if not ((overlap_min_u <= overlap_max_u) and (overlap_min_v <= overlap_max_v)):
        return [0, 0] # No overlap in 2D, they cannot collide.
    
    # Check COAABB overlap
    a_lims = [(a_min_u, a_max_u), (a_min_v, a_max_v)]
    b_lims = [(b_min_u, b_max_u), (b_min_v, b_max_v)]
    coaabb_overlap = check_COAABB_overlap(a_lims, b_lims)

    if not coaabb_overlap:
        # AABBs overlap roughly, but the tighter COAABBs missed each other. Safe.
        return [0, 0] 
    
    # Static Overlap Check: If they are clashing right now, it blocks both directions!
    if a_max_w >= b_min_w and a_min_w <= b_max_w:
        return [2, 2]  # Hard static collision! Blocked.

    # Directional macro-blocking check
    if a_min_w >= b_max_w :
        result[1] = 2 # Blocked in negative direction
    if b_min_w >= a_max_w:
        result[0] = 2  # Blocked in positive direction

    # Return the result of the checks
    return result # If it hasn't returned yet, it means the macro shapes overlap in 2D, but their 
                  # depths are inconclusive (we need to run the facet intersection tests)

def focus_facet_intersection_test(pf_a: PseudoFace, pf_b: PseudoFace, direction: str):
    """Checks if any of the focus facets of PseudoFace of part A intersects with any of those of part B.
    Direction is either '+w' or '-w' depending on whether we are checking the positive or negative extraction direction
    Returns:
        0 if no collision detected between any of the focus facets.
        1 if A cannot be extracted in the positive direction without colliding with B, but can be extracted in the negative direction.
        -1 if A cannot be extracted in the negative direction without colliding with B, but can be extracted in the positive direction"""
    
    # First we test the AABBs of the candidates in 2D
    for facet_a in pf_a.focus_facets:
        coords_2d_a = pf_a.triangles_2d[facet_a]
        min_u_a, min_v_a = coords_2d_a.min(axis=0)
        max_u_a, max_v_a = coords_2d_a.max(axis=0)

        for facet_b in pf_b.focus_facets:
            coords_2d_b = pf_b.triangles_2d[facet_b]
            min_u_b, min_v_b = coords_2d_b.min(axis=0)
            max_u_b, max_v_b = coords_2d_b.max(axis=0)

            # 1. Check if the AABBs of the facets overlap in 2D
            # If the boxes don't overlap in U or V, they can't touch!
            if (min_u_a > max_u_b or max_u_a < min_u_b or
                min_v_a > max_v_b or max_v_a < min_v_b):
                continue # Skip to the next pair instantly

            # 2. Check 2D polygon intersection
            poly_a = Polygon(coords_2d_a)
            poly_b = Polygon(coords_2d_b)

            if not poly_a.intersects(poly_b):
                continue # If the 2D projections don't intersect, skip to the next pair

            # 3. If theres a 2D intersection we check the depth to see if they collide in +/- w
            extraction_axis = pf_a.extraction_axis
            min_w_a = pf_a.triangles_3d[facet_a][:, extraction_axis].min()
            max_w_a = pf_a.triangles_3d[facet_a][:, extraction_axis].max()
            min_w_b = pf_b.triangles_3d[facet_b][:, extraction_axis].min()
            max_w_b = pf_b.triangles_3d[facet_b][:, extraction_axis].max()

            # Case a: Static overlap
            if max_w_a >= min_w_b and min_w_a <= max_w_b:
                return 2 # Collide instantly in both directions
            
            # Case b: I need to extract A in the positive direction, so I check if B is blocking that
            elif max_w_a <= min_w_b and direction == "+w":
                return 1 # A cannot be extracted in the positive direction without colliding with B
        
            # Case c: I need to extract A in the negative direction, so I check if B is blocking that
            elif min_w_a >= max_w_b and direction == "-w":
                return -1 # A cannot be extracted in the negative direction without colliding with B
            
    return 0 # No collision detected between any of the focus facets

def focus_facet_intersection_full(pseudo_faces_a, pseudo_faces_b):
    extraction_axis = pseudo_faces_a[0].extraction_axis
    pos_result = 0
    neg_result = 0
    for pf_a in pseudo_faces_a:
        for pf_b in pseudo_faces_b:
                ff_intersection_pos = focus_facet_intersection_test(pf_a, pf_b, "+w")
                ff_intersection_neg = focus_facet_intersection_test(pf_a, pf_b, "-w")

                if ff_intersection_pos == 2 or ff_intersection_neg == 2:
                    return 1, 1
                else:
                    if ff_intersection_pos == 1:
                        pos_result = 1
                    elif ff_intersection_neg == 1:
                        neg_result = 1
    return pos_result, neg_result


## Determine if parts intersect their AABBs
def check_3D_AABB_intersection(part_a, part_b):
    a_min_3d = part_a.bounds[0]
    a_max_3d = part_a.bounds[1]
    b_min_3d = part_b.bounds[0]
    b_max_3d = part_b.bounds[1]

    # Evaluate intersection boolean
    intersects = not (np.any(a_min_3d > b_max_3d) or np.any(a_max_3d < b_min_3d))
    
    # Always return the tuples, just flip the boolean flag!
    return [(a_min_3d, a_max_3d), (b_min_3d, b_max_3d), intersects]


## Narrow Phase Test functions (facet intersection)
def check_static_interference(part_a, part_b):
    "Checks if part_a and part_b are already colliding in their current position, means that they statically interfere"
    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object('part_a', part_a)
    collision_manager.add_object('part_b', part_b)

    is_colliding = collision_manager.in_collision_internal()
    return is_colliding

def filter_facets(pf_a: PseudoFace, pf_b: PseudoFace, AABB_3d_intersection, only_focus_facets = False, tolerance = 1e-4):
    w_idx = pf_a.extraction_axis # Same for both pf_a or pf_b
    a_min_3d, a_max_3d = AABB_3d_intersection[0]
    b_min_3d, b_max_3d = AABB_3d_intersection[1]
    parts_intersect = AABB_3d_intersection[2]
    
    w1_max = b_max_3d[w_idx]
    w0_min = a_min_3d[w_idx]

    candidates_a = []
    candidates_b = []

    list_to_check_a = pf_a.focus_facets if only_focus_facets else pf_a.triangles_3d
    list_to_check_b = pf_b.focus_facets if only_focus_facets else pf_b.triangles_3d

    # --- PART A LOOP ---
    for idx in range(len(list_to_check_a)):
        local_idx = idx if not only_focus_facets else pf_a.focus_facets[idx]
        facet_a = pf_a.triangles_3d[local_idx]
        min_w = facet_a[:, w_idx].min()
        max_w = facet_a[:, w_idx].max()
        normal_w = pf_a.part.face_normals[local_idx][w_idx]

        # If the parts dont intersect we check those with normals pointing towards the extraction direction
        if not parts_intersect:
            if abs(normal_w) > 0:
                candidates_a.append(local_idx)
        
        # If they do intersect we filter according to facet normals
        else:
            if max_w >= w1_max and normal_w > 0:  # <--- FIXED TARGET (w1_max)
                candidates_a.append(local_idx)
            elif min_w < w1_max and normal_w < 0:
                candidates_a.append(local_idx)

    # --- PART B LOOP ---
    for idx in range(len(list_to_check_b)):       # <--- FIXED LOOP VARIABLE (idx)
        local_idx = idx if not only_focus_facets else pf_b.focus_facets[idx]
        
        facet_b = pf_b.triangles_3d[local_idx]
        min_w = facet_b[:, w_idx].min()
        max_w = facet_b[:, w_idx].max()
        normal_w = pf_b.part.face_normals[local_idx][w_idx]

        # If the parts dont intersect we check those with normals pointing towards the extraction direction
        if not parts_intersect:
            if abs(normal_w) > 0:
                candidates_b.append(local_idx)
        
        # If they do intersect we filter according to facet normals
        else:
            if max_w > w0_min and normal_w > 0:
                candidates_b.append(local_idx)
            elif min_w <= w0_min and normal_w < 0:
                candidates_b.append(local_idx)

    return candidates_a, candidates_b

def hybrid_facet_intersection_test(part_a, part_b, facet_a, facet_b, use_MRT, MRT_tolerance = 1e-4):
    # 1. Check AABB of facets in 3D to discard impossible pairs instantly
    a_min = facet_a.min(axis=0)
    a_max = facet_a.max(axis=0)
    b_min = facet_b.min(axis=0)
    b_max = facet_b.max(axis=0)

    if (a_min[0] > b_max[0] or a_max[0] < b_min[0] or
        a_min[1] > b_max[1] or a_max[1] < b_min[1] or
        a_min[2] > b_max[2] or a_max[2] < b_min[2]):
        return 0 # If the boxes don't overlap in any dimension, they can't touch!
    
    # Revise specific indexing since i dont yet know how to input the facets
    poly_a = Polygon(facet_a[:, :2]) # Project to 2D (U and V)
    poly_b = Polygon(facet_b[:, :2]) # Project to 2D (U and V)
    
    # 3. Case 1: If there is static interference we use MRT
    if use_MRT:
        overlap_poly = poly_a.intersection(poly_b)
        min_u, min_v, max_u, max_v = overlap_poly.bounds

        overlap_width = max_u - min_u
        overlap_height = max_v - min_v

        overlap_distance = min(overlap_width, overlap_height)
         
        if overlap_distance < MRT_tolerance:
            return 1 # If the overlapping area is very small, we consider it a minor interference (1)
        
        return 2
    
    # 4. Case 2: If there is no static interference we check with standard collision tests with polygon intersection
    if not poly_a.intersects(poly_b):
        return 0 # If the 2D projections don't intersect, they can't collide
    return 2

def get_primitive_points(poly_a: Polygon, poly_b: Polygon):
    if not poly_a.intersects(poly_b):
            return np.empty((0, 2))

    overlap = poly_a.intersection(poly_b)
    raw_coords = []

    # Case A: Standard single overlapping polygon area
    if isinstance(overlap, Polygon):
        raw_coords.extend(list(overlap.exterior.coords)[:-1])

    # Case B: Multiple separated overlapping areas (MultiPolygon)
    elif isinstance(overlap, MultiPolygon):
        for poly in overlap.geoms:
            raw_coords.extend(list(poly.exterior.coords)[:-1])

    # Case C: Lower-dimension contacts (LineString, Point, or Collections)
    else:
        # If it's a line touch or vertex point touch, extract coordinates directly
        if hasattr(overlap, 'coords'):
            raw_coords.extend(list(overlap.coords))
        elif hasattr(overlap, 'geoms'):
            for geom in overlap.geoms:
                if hasattr(geom, 'coords'):
                    raw_coords.extend(list(geom.coords))

    # Remove any duplicate coordinate entries to keep the points unique
    if len(raw_coords) > 0:
        unique_pts = np.unique(np.array(raw_coords), axis=0)
    
    # Project these unique points onto the facet
        
    return np.empty((0, 2))

def primitive_point_projection(pf, facet_idx, primitive_points):
    # Solve equation of type A*u + B*v * C*w + D = 0 so we can extract w coordinate of
    # primitive points projected on the facet plane

    # Obtain pseudoface normals on each axis
    nu = pf.part.face_normals[facet_idx][pf.u_axis]
    nv = pf.part.face_normals[facet_idx][pf.v_axis]
    nw = pf.part.face_normals[facet_idx][pf.extraction_axis]

    # Get any point on the facet (e.g., the first vertex of the triangle)
    u0 = pf.triangles_3d[facet_idx][0, pf.u_axis]
    v0 = pf.triangles_3d[facet_idx][0, pf.v_axis]
    w0 = pf.triangles_3d[facet_idx][0, pf.extraction_axis]

    # Solve linear plane equation
    D = -(nu * u0 + nv * v0 + nw * w0)
    projected_w = -(nu * primitive_points[:, 0] + nv * primitive_points[:, 1] + D) / nw

    # Return 3D coordinates of projected points
    projected_points_3d = np.zeros((primitive_points.shape[0], 3))
    projected_points_3d[:, pf.u_axis] = primitive_points[:, 0]
    projected_points_3d[:, pf.v_axis] = primitive_points[:, 1]
    projected_points_3d[:, pf.extraction_axis] = projected_w

    return projected_points_3d

def IM_entry_calculation(pf_a, facet_idx_a, pf_b, facet_idx_b, primitive_points_a, primitive_points_b,
                         interference_type):
    offset_tolerance = 1e-4
    a_ij = 0
    a_ji = 0

    # Get 3d w bounds of the facets
    w_bounds_a = pf_a.triangles_3d[facet_idx_a][:, pf_a.extraction_axis]
    w_bounds_b = pf_b.triangles_3d[facet_idx_b][:, pf_b.extraction_axis]

    w_max_a = max(w_bounds_a)
    w_min_a = min(w_bounds_a)
    w_max_b = max(w_bounds_b)
    w_min_b = min(w_bounds_b)

    # Get normal components in the extraction direction
    normal_w_a = pf_a.part.face_normals[facet_idx_a][pf_a.extraction_axis]
    normal_w_b = pf_b.part.face_normals[facet_idx_b][pf_b.extraction_axis]

    entry_dict = {"minor": 1, "major": 2}
    entry_val = interference_type


    # check if the primitive points of A are same as those of B
    # pick any 2 vertexes of the facets and check if dot product of difference vector with the normal is 0
    diff_vector = pf_a.triangles_3d[facet_idx_a][0] - pf_b.triangles_3d[facet_idx_b][0]
    offset = np.dot(diff_vector, pf_a.part.face_normals[facet_idx_a])
    # check if facets are flush and parallel (same normals and zero offset)
    same_plane = abs(offset) < offset_tolerance and abs(np.dot(diff_vector, pf_b.part.face_normals[facet_idx_b])) < offset_tolerance


    if w_max_a <= w_min_b and w_min_a != w_max_b:
        a_ij = entry_val # A is fully on the negative extraction side of B, but they are not perfectly flush (which would be a static interference)

    elif w_max_b <= w_min_a and w_max_a != w_min_b:
        a_ji = entry_val # B is fully on the negative extraction side of A, but they are not perfectly flush (which would be a static interference)
    
    elif not same_plane:
        for i in range(primitive_points_a.shape[0]):
            w_prim_a = primitive_points_a[i][pf_a.extraction_axis]
            w_prim_b = primitive_points_b[i][pf_b.extraction_axis]

            if abs(w_prim_a - w_prim_b) < 1e-3:
                continue 
            elif w_prim_a < w_prim_b:
                a_ij = entry_val # Primitive point of A is on the negative extraction side of B
            else:                
                a_ji = entry_val # Primitive point of B is on the negative extraction side of A
            
            # Check if both entries are != 0 so we stop checking further
            if a_ij != 0 and a_ji != 0:
                break
            
    else:
        if (normal_w_a > 0 and normal_w_b > 0) :
            a_ij = entry_val # They are parallel and facing the same direction, so we consider it a minor interference (1)
        elif (normal_w_a < 0 and normal_w_b < 0):
            a_ji = entry_val # They are parallel and facing the same direction, so we consider it a minor interference (1)
        else:
            a_ij = entry_val # They interfere for extraction of both parts in the same direction
            a_ji = entry_val
    
    return a_ij, a_ji

def evaluate_narrow_phase(candidates_a, candidates_b, pf_a, pf_b, part_a_aux, part_b_aux, use_MRT):
    """Evaluates pairs of candidate triangles and returns the maximum directional interference."""
    max_pos = 0
    max_neg = 0
    
    # product() flattens the double loop into ONE level of indentation!
    for idx_a, idx_b in product(candidates_a, candidates_b):
        facet_a = pf_a.triangles_3d[idx_a]
        facet_b = pf_b.triangles_3d[idx_b]
        
        hybrid_result = hybrid_facet_intersection_test(
            part_a_aux, part_b_aux, facet_a, facet_b, use_MRT
        )

        if hybrid_result not in [1, 2]:
            continue

        primitive_all = get_primitive_points(facet_a, facet_b)
        primitive_points_a = primitive_point_projection(pf_a, idx_a, primitive_all)
        primitive_points_b = primitive_point_projection(pf_b, idx_b, primitive_all)
        
        positive_entry, negative_entry = IM_entry_calculation(
            pf_a, idx_a, pf_b, idx_b, primitive_points_a, primitive_points_b, hybrid_result
        )

        # Track the worst-case collision found so far
        max_pos = max(max_pos, positive_entry)
        max_neg = max(max_neg, negative_entry)

        # If both directions are fully blocked, stop checking triangles!
        if max_pos == 2 and max_neg == 2:
            break 
            
    return max_pos, max_neg


## Main Extraction functions
def evaluate_pair_interference(part_a_data, part_b_data, extraction_axis):
    """Evaluates the maximum interference between two parts along a specific axis."""
    # Unpack the pre-calculated data!
    part_a = part_a_data["part_mesh"]
    part_b = part_b_data["part_mesh"]
    to_origin_A = part_a_data["to_origin"]
    
    part_a_aux, part_b_aux = part_a.copy(), part_b.copy()
    part_a_aux.apply_transform(to_origin_A)
    part_b_aux.apply_transform(to_origin_A)

    parts_AABB_interfere = check_3D_AABB_intersection(part_a_aux, part_b_aux)
    use_MRT = check_static_interference(part_a_aux, part_b_aux)
    if parts_AABB_interfere[2] == True:
        use_MRT = True

    overlap_region, overlap_result = check_2d_aabb_overlap(
        part_a_aux.bounding_box.bounds, part_b_aux.bounding_box.bounds, extraction_axis
    )
    
    # Return immediately if the broad phase gives a definitive answer
    if overlap_result == 0: return 0, 0
    if overlap_result == -1: return 0, 2
    if overlap_result == 1: return 2, 0
    if overlap_result == 2: return 2, 2

    # 2. PseudoFace Generation
    pseudo_faces_a = create_PFs(part_a_aux, extraction_axis)
    pseudo_faces_b = create_PFs(part_b_aux, extraction_axis)
    for pf_a in pseudo_faces_a: pf_a.get_focus_facets(overlap_region)
    for pf_b in pseudo_faces_b: pf_b.get_focus_facets(overlap_region)

    max_pos, max_neg = 0, 0
    full_interference = False
    
    for pf_a, pf_b in product(pseudo_faces_a, pseudo_faces_b):
        if full_interference: break
            
        pf_intersect = check_PF_overlap(pf_a, pf_b)
        final_pos, final_neg = pf_intersect

        if -2 in pf_intersect:
            for attempt in ["focus_facets", "full_fallback"]:
                is_focus = (attempt == "focus_facets")
                candidates_a, candidates_b = filter_facets(
                    pf_a, pf_b, parts_AABB_interfere, only_focus_facets=is_focus
                )
                
                if not candidates_a or not candidates_b: continue 

                c_pos, c_neg = evaluate_narrow_phase(
                    candidates_a, candidates_b, pf_a, pf_b, part_a_aux, part_b_aux, use_MRT
                )
                
                final_pos, final_neg = max(final_pos, c_pos), max(final_neg, c_neg)
                if final_pos == 2 and final_neg == 2: break 

        max_pos, max_neg = max(max_pos, final_pos), max(max_neg, final_neg)
        if max_pos == 2 and max_neg == 2:
            full_interference = True
            break

    return max_pos, max_neg

def calculate_IM_matrices(assembly_manifest):
    N = len(assembly_manifest)
    matrices = {d: np.zeros((N, N), dtype=int) for d in ["+x", "-x", "+y", "-y", "+z", "-z"]}
    axis_matrix_map = {"x": ["+x", "-x"], "y": ["+y", "-y"], "z": ["+z", "-z"]}
    part_keys = list(assembly_manifest.keys())

    for extraction_axis, (pos_key, neg_key) in axis_matrix_map.items():
        print(f'\n----------------- Checking {extraction_axis} Direction -----------------')
        
        for i, j in permutations(range(N), 2):
            part_a_name = part_keys[i]
            part_b_name = part_keys[j]
            
            part_a_data = assembly_manifest[part_a_name]
            part_b_data = assembly_manifest[part_b_name]

            # Names perfectly synced with the updated helper function
            pos_val, neg_val = evaluate_pair_interference(
                part_a_data, part_b_data, extraction_axis)
            # pos_val, neg_val = evaluate_pair_interference(
            #     part_a_data, part_b_data, extraction_axis, part_a_name, part_b_name
            # )
            
            matrices[pos_key][i, j] = pos_val
            matrices[neg_key][i, j] = neg_val

    return matrices

## Data Handling
def clean_obb_matrix(to_origin, tolerance=0.05):
    """
    Snaps microscopic noise to 0/1, then uses Singular Value Decomposition (SVD) 
    to guarantee the resulting matrix is a perfectly orthogonal 3D rotation matrix.
    """
    matrix = to_origin.copy()
    rot = matrix[:3, :3]
    
    # 1. Snap the microscopic noise on intended flush axes
    rot[np.abs(rot) < tolerance] = 0.0
    rot[np.abs(rot - 1.0) < tolerance] = 1.0
    rot[np.abs(rot + 1.0) < tolerance] = -1.0
    
    # 2. SVD Re-Orthogonalization 
    # This takes the snapped matrix and mathematically forces the axes to be exactly 
    # 90 degrees apart and length 1.0, completely preventing CAD mesh warping!
    U, _, Vt = np.linalg.svd(rot)
    perfect_rot = np.dot(U, Vt)
    
    # 3. Failsafe: Ensure it's a true rotation (determinant of +1) and not a reflection
    if np.linalg.det(perfect_rot) < 0:
        Vt[2, :] *= -1
        perfect_rot = np.dot(U, Vt)
        
    matrix[:3, :3] = perfect_rot
    return matrix

def load_assembly_from_folder(folder_path):
    assembly_manifest = {}
    matrix_idx = 0
    
    # 1. Gather and sort all STL files in the directory alphabetical order
    folder = Path(folder_path)
    stl_files = sorted(list(folder.glob("*.stl")))
    
    # 2. Iterate through the sorted files to build your dictionary
    for file_path in stl_files:
        raw_name = file_path.stem  # e.g., "Ensamblaje1 - Lid-1"
        
        # Split the string at the hyphen and keep only the last part
        if " - " in raw_name:
            part_name = raw_name.split(" - ")[-1].strip() # Becomes "Lid-1"
        else:
            part_name = raw_name
        
        # Load the mesh geometry
        mesh_geom = trimesh.load(str(file_path))

        mesh_geom.merge_vertices()
        
        # Get the Oriented Bounding Box transformation matrix
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh_geom)
        
        # ---> Clean the matrix and re-orthogonalize it! <---
        to_origin = clean_obb_matrix(to_origin)
        
        # Because the matrix is now mathematically perfect, the inverse will be flawless
        from_origin = np.linalg.inv(to_origin)

        # Structure the inner data dictionary
        extraction_vectors = {
            "+x": from_origin[:3, 0],
            "-x": -from_origin[:3, 0],
            "+y": from_origin[:3, 1],
            "-y": -from_origin[:3, 1],
            "+z": from_origin[:3, 2],
            "-z": -from_origin[:3, 2]
        }

        # Structure the inner data dictionary
        assembly_manifest[part_name] = {
            "matrix_idx": matrix_idx,
            "part_mesh": mesh_geom,
            "to_origin": to_origin,              # Store this to speed up the main loop!
            "extraction_vectors": extraction_vectors,
            "center_point": from_origin[:3, 3]   # The exact center of the OBB
        }
        
        # Move to the next index slot
        matrix_idx += 1
        
    return assembly_manifest

def export_matrices_to_excel(matrices, assembly_manifest, output_folder="output_matrices", filename="Interference_Matrices.xlsx"):
    """
    Exports the 6 directional matrices to a single Excel file, 
    putting each matrix on its own named tab.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)
    
    # Extract part names sorted by their matrix_idx to guarantee exact alignment
    sorted_parts = sorted(assembly_manifest.items(), key=lambda x: x[1]["matrix_idx"])
    part_names = [name for name, data in sorted_parts]

    # Open the Excel writer engine
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for direction, matrix in matrices.items():
            
            # Create the DataFrame
            df = pd.DataFrame(matrix, index=part_names, columns=part_names)
            df.index.name = "Moving \ Stationary"
            
            # Create a clean tab name (e.g., "+x" becomes "Pos_X")
            tab_name = direction.replace("+", "Pos_").replace("-", "Neg_").upper()
            
            # Write this specific matrix to its own tab
            df.to_excel(writer, sheet_name=tab_name)
            
    print(f"Successfully saved all matrices to a single Excel file: {filepath}")

def export_matrices_to_csv(matrices, assembly_manifest, output_folder="output_matrices"):
    """
    Exports the 6 directional matrices to CSV files with part names as headers.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract part names sorted by their matrix_idx to guarantee exact alignment
    sorted_parts = sorted(assembly_manifest.items(), key=lambda x: x[1]["matrix_idx"])
    part_names = [name for name, data in sorted_parts]

    # Export each direction as its own spreadsheet
    for direction, matrix in matrices.items():
        # Create a Pandas DataFrame to bind the matrix to the part names
        df = pd.DataFrame(matrix, index=part_names, columns=part_names)
        
        # Save it cleanly to disk
        safe_dir_name = direction.replace("+", "Pos_").replace("-", "Neg_")
        filepath = os.path.join(output_folder, f"IM_Matrix_{safe_dir_name}.csv")
        
        df.to_csv(filepath)
        print(f"Successfully saved {direction} matrix to: {filepath}")             

def export_directions_to_excel(assembly_manifest, output_folder="output_matrices", filename="Robot_Extraction_Vectors.xlsx"):
    """
    Exports the 3D extraction vectors for each part to an Excel sheet.
    """
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)
    
    # Prepare the data dictionary for Pandas
    data = {"Part Name": []}
    directions = ["+x", "-x", "+y", "-y", "+z", "-z"]
    for d in directions:
        data[d] = []
        
    for part_name, properties in assembly_manifest.items():
        data["Part Name"].append(part_name)
        for d in directions:
            # Format the vector as a clean, rounded string: "[1.000, 0.000, 0.000]"
            v = properties["extraction_vectors"][d]
            data[d].append(f"[{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")
            
    # Export to Excel
    df = pd.DataFrame(data)
    df.set_index("Part Name", inplace=True)
    df.to_excel(filepath)
    print(f"Successfully saved Robot Extraction Vectors to: {filepath}")

## Auxiliary visualization Functions with pyvista
def visualize_pseudofaces(part, pseudo_faces_list):
    """
    Renders the full part in transparent gray, and paints each 
    Pseudo Face object a different solid color.
    """
    # 1. Convert the trimesh object to a pyvista object
    mesh = pv.wrap(part)
    
    # 2. Set up the 3D window
    pl = pv.Plotter()
    
    # 3. Draw the original part as a faint "ghost" for context
    pl.add_mesh(mesh, color='white', opacity=0.15)
    
    # 4. A list of bright colors to cycle through
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange']
    
    # 5. Loop through your instantiated PseudoFace objects
    for i, pf in enumerate(pseudo_faces_list):
        # Extract only the triangles that belong to this PF!
        # Because we converted face_indices to a numpy array, this works instantly.
        pf_mesh = mesh.extract_cells(pf.face_indices)
        
        # Pick a color (loops back to the start if you have more than 7 PFs)
        c = colors[i % len(colors)]
        
        # Draw this specific PF solid and show its black triangle edges
        pl.add_mesh(pf_mesh, color=c, show_edges=True, line_width=1)
        
    # Show the interactive window!
    pl.show()

def visualize_part_axes(part_name, assembly_manifest):
    """
    Visualizes a specific part in its original global position 
    and draws its extraction axes based on its OBB.
    """
    properties = assembly_manifest[part_name]
    mesh = properties["part_mesh"]
    center = properties["center_point"]
    vectors = properties["extraction_vectors"]
    
    plotter = pv.Plotter(title=f"Extraction Axes: {part_name}")
    plotter.add_mesh(pv.wrap(mesh), color="lightgray", opacity=0.8, show_edges=True)

    # Add an arrow for the Local +X axis (Red)
    arrow_x = pv.Arrow(start=center, direction=vectors["+x"], scale=15)
    plotter.add_mesh(arrow_x, color='red')

    # Add an arrow for the Local +Y axis (Green)
    arrow_y = pv.Arrow(start=center, direction=vectors["+y"], scale=15)
    plotter.add_mesh(arrow_y, color='green')

    # Add an arrow for the Local +Z axis (Blue)
    arrow_z = pv.Arrow(start=center, direction=vectors["+z"], scale=15)
    plotter.add_mesh(arrow_z, color='blue')

    # Add a tiny sphere at the center point so we can see the origin of the arrows
    plotter.add_mesh(pv.Sphere(radius=1.5, center=center), color='black')

    plotter.show()

def visualize_narrow_phase(pseudo_faces, overlap_region, plotter, index, show = False):
    
    for i, pf in enumerate(pseudo_faces):
        if i == 0:
            pf.visualize_focus_facets(overlap_region, plotter, index)
        else:
            pf.visualize_focus_facets(overlap_region, plotter, index, show_SR_box=False)
    if show:
        plotter.show()
# ----------------------------------------------------- COMPLEMENTARY FUNCTIONS ---------------------------------------------- 

# For the loop i need to revert the transformation applied to part_b so i
# can get extraction directions in the original frame
if __name__ == "__main__":
    assembly_manifest = load_assembly_from_folder('STLs/EndEffector')
    
    # 1. VISUALIZE FIRST! 
    # Grab the name of the first part and visualize its axes
    first_part = list(assembly_manifest.keys())[0]
    visualize_part_axes(first_part, assembly_manifest)
    
    # 2. RUN THE HEAVY MATH
    start_time = time.time()
    final_matrices = calculate_IM_matrices(assembly_manifest)
    export_matrices_to_excel(final_matrices, assembly_manifest)
    print(f"--- Time Taken: {(time.time() - start_time):.2f} seconds ---")

    # 3. EXPORT THE ROBOT VECTORS
    export_directions_to_excel(assembly_manifest)




