import trimesh
import pyvista as pv
import numpy as np
import networkx as nx
from classes import PseudoFace
from shapely.geometry import Polygon

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

    a_min_w, a_max_w = bounds_a[0][extraction_axis], bounds_a[1][extraction_axis]
    b_min_w, b_max_w = bounds_b[0][extraction_axis], bounds_b[1][extraction_axis]

    # Case 3: AABBs overlap and COAABBs overlap
    overlap_result = None
    if a_min_w >= b_max_w:
         overlap_result = -1 # Part A can be extracted in extraction direction without colliding with B, 
                                       # but not in the opposite direction
    elif b_min_w >= a_max_w:
        overlap_result = 1   # Part A cannot be extracted in extraction direction without colliding with B, 
                                       # but can be extracted in the opposite direction
    else:
        overlap_result = 2   # Part A cannot be extracted in either direction without colliding with B

    return (overlap_region, overlap_result)

    # Note: The return values are as follows:
    #  0: No overlap at all (AABBs don't even touch)
    # -2: AABBs overlap but COAABBs do not (We need to check PFs)
    # -1: A can be extracted in the positive extraction direction without colliding with B
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
def filter_facets(part, extraction_axis, overlap_region, tolerance = 1e-4):
    # Figure out which two axes form our 2D "shadow" plane
    # If we extract in Z (2), our 2D plane uses X (0) and Y (1).
    axis_idx = {"x": 0, "y": 1, "z": 2}
    all_axes = [0, 1, 2]
    all_axes.remove(axis_idx[extraction_axis])
    u_axis = all_axes[0]
    v_axis = all_axes[1]

    # Get face normals for the part in line with the extraction axis
    normals_w = part.face_normals[:, axis_idx[extraction_axis]]
    #print(normals_w)
    valid_faces_mask = np.abs(normals_w) > tolerance
    #print(valid_faces_mask)

    # Get triangles of the part (indexes are [face index, vertex index, coordinate index])
    triangles_u = part.triangles[:, :, u_axis]
    triangles_v = part.triangles[:, :, v_axis]

    facet_min_u, facet_max_u = np.min(triangles_u, axis=1), np.max(triangles_u, axis=1)
    facet_min_v, facet_max_v = np.min(triangles_v, axis=1), np.max(triangles_v, axis=1)

    overlap_u_min, overlap_u_max = overlap_region['overlap_u']
    overlap_v_min, overlap_v_max = overlap_region['overlap_v']

    # Check if the facet's projection overlaps with the overlap region
    in_sr_u = (facet_min_u <= overlap_u_max) & (facet_max_u >= overlap_u_min)
    in_sr_v = (facet_min_v <= overlap_v_max) & (facet_max_v >= overlap_v_min)

    valid_faces_mask = valid_faces_mask & in_sr_u & in_sr_v
    valid_face_indices = np.where(valid_faces_mask)[0]

    return valid_face_indices

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

def check_PF_overlap(pf_a, pf_b):
    ## Need to implement
    return

## Facet projection intersection test functions
def focus_facet_intersection_test(pf_a: PseudoFace, pf_b: PseudoFace, direction: str):
    "Checks if any of the focus facets of PseudoFace of part A intersects with any of those of part B"
    "Direction is either '+w' or '-w' depending on whether we are checking the positive or negative extraction direction"
    "Returns:"
    "   0 if no collision detected between any of the focus facets"
    "   1 if A cannot be extracted in the positive direction without colliding with B, but can be extracted in the negative direction"
    "  -1 if A cannot be extracted in the negative direction without colliding with B, but can be extracted in the positive direction"
    
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
            if max_w_a <= min_w_b and direction == "+w":
                return 1 # A cannot be extracted in the positive direction without colliding with B
        
            # Case c: I need to extract A in the negative direction, so I check if B is blocking that
            if min_w_a >= max_w_b and direction == "-w":
                return -1 # A cannot be extracted in the negative direction without colliding with B
            
    return 0 # No collision detected between any of the focus facets

## Main extraction check function
def main_extraction_check(part_a, part_b,):
    # Get the solid bounding boxes
    bbox_a = part_a.bounding_box
    bbox_b = part_b.bounding_box

    # Get Oriented bounding box with respect to part a
    to_origin_A, extents_A = trimesh.bounds.oriented_bounds(part_a)
    from_origin_A = np.linalg.inv(to_origin_A)


    return

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

def visualize_extraction_directions(part_a, part_b, local_x_dir = [1, 0, 0], local_y_dir = [0, 1, 0], local_z_dir = [0, 0, 1], center_point = [0, 0, 0]):
    plotter = pv.Plotter()
    plotter.add_mesh(pv.wrap(part_a), color="lightgray", opacity=0.8)
    plotter.add_mesh(pv.wrap(part_b), color="lightblue", opacity=0.8)

    # Add an arrow for the Local X-axis (Red)
    arrow_x = pv.Arrow(start=center_point, direction=local_x_dir, scale=10)
    plotter.add_mesh(arrow_x, color='red')

    # Add an arrow for the Local Y-axis (Green)
    arrow_y = pv.Arrow(start=center_point, direction=local_y_dir, scale=10)
    plotter.add_mesh(arrow_y, color='green')

    # Add an arrow for the Local Z-axis (Blue)
    arrow_z = pv.Arrow(start=center_point, direction=local_z_dir, scale=10)
    plotter.add_mesh(arrow_z, color='blue')

    # Show the interactive window
    plotter.show()

def visualize_narrow_phase(pseudo_faces, overlap_region):
    plotter = pv.Plotter()
    for i, pf in enumerate(pseudo_faces):
        if i == 0:
            pf.visualize_focus_facets(overlap_region, plotter)
        else:
            pf.visualize_focus_facets(overlap_region, plotter, show_SR_box=False)
    plotter.show()
# ----------------------------------------------------- COMPLEMENTARY FUNCTIONS ---------------------------------------------- 

# For the loop i need to revert the transformation applied to part_b so i
# can get extraction directions in the original frame
if __name__ == "__main__":

    # Load models using trimesh
    part_a = trimesh.load('STLs/Ensamblaje1 - Lid-1.STL')
    part_b = trimesh.load('STLs/Ensamblaje1 - Pen-1.STL')

    # Get Oriented bounding box with respect to part a
    to_origin_A, extents_A = trimesh.bounds.oriented_bounds(part_a)
    from_origin_A = np.linalg.inv(to_origin_A)

    # Get directions of the OBB axes for part_a
    local_x_dir = from_origin_A[:3, 0]
    local_y_dir = from_origin_A[:3, 1]
    local_z_dir = from_origin_A[:3, 2]
    center_point = from_origin_A[:3, 3]

    # Create auxiliary copies (to avoid modifying the original meshes)
    part_a_aux = part_a.copy()
    part_b_aux = part_b.copy()

    # Apply transformation to align part_a with the world axes (so its OBB becomes an AABB)
    part_a_aux.apply_transform(to_origin_A)
    part_b_aux.apply_transform(to_origin_A)

    # Get the solid bounding boxes
    bbox_a = part_a_aux.bounding_box
    bbox_b = part_b_aux.bounding_box

    # --- Let's test it on your data! ---
    visualize_extraction_directions(part_a_aux, part_b_aux)
    axis_test = input("Enter the extraction axis (x, y, or z): ").lower()

    # 1. Check AABB and COAABB overlap
    overlap_region, overlap_result = check_2d_aabb_overlap(bbox_a.bounds, bbox_b.bounds, extraction_axis=axis_test)

    # Test Pseudo Face creation and visualization
    pseudo_faces_a = create_PFs(part_a_aux, extraction_axis=axis_test)
    pseudo_faces_b = create_PFs(part_b_aux, extraction_axis=axis_test)

    for pf in pseudo_faces_a:
        pf.get_focus_facets(overlap_region)
    for pf in pseudo_faces_b:
        pf.get_focus_facets(overlap_region)

    visualize_narrow_phase(pseudo_faces_a, overlap_region)
    visualize_narrow_phase(pseudo_faces_b, overlap_region)



