import trimesh
import pyvista as pv
import numpy as np
# ----------------------------------------------------- MAIN FUNCTIONS ----------------------------------------------
## AABB overlap test functions

def check_2d_aabb_overlap(bounds_a, bounds_b, extraction_axis):
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

    # If AABBs dont overlap then they dont collide so we return immediately
    if not ((overlap_min_u <= overlap_max_u) and (overlap_min_v <= overlap_max_v)):
        return 0  # No overlap
    
    # AABBs overlap, now we check COAABB overlap
    a_lims = [(a_min_u, a_max_u), (a_min_v, a_max_v)]
    b_lims = [(b_min_u, b_max_u), (b_min_v, b_max_v)]
    coaabb_overlap = check_COAABB_overlap(a_lims, b_lims)

    # If the COAABBs don't overlap, we can return -2 to indicate that we need to check the PFs
    if not coaabb_overlap:
        return -2 

    a_min_w, a_max_w = bounds_a[0][extraction_axis], bounds_a[1][extraction_axis]
    b_min_w, b_max_w = bounds_b[0][extraction_axis], bounds_b[1][extraction_axis]

    if a_min_w >= b_max_w:
        return -1 # Part A can be extracted in extraction direction without colliding with B, 
                  # but not in the opposite direction
    elif b_min_w >= a_max_w:
        return 1 # Part A cannot be extracted in extraction direction without colliding with B, 
                 # but can be extracted in the opposite direction
    else:
        return 2 # Part A cannot be extracted in either direction without colliding with B
    
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
def filter_facets(bounds_a, bounds_b):
    ## Need to implement
    return

def create_PFs(bounds_a, bounds_b):
    ## Need to implement
    return

def check_PF_overlap(pf_a, pf_b):
    ## Need to implement
    return

## Facet projection intersection test functions
def check_facet_intersection(part_a, part_b):
    ## Need to implement
    return


# ----------------------------------------------------- COMPLEMENTARY FUNCTIONS ---------------------------------------------- 


if __name__ == "__main__":

    # Load models using trimesh
    part_a = trimesh.load('STLs/Test Assembly - Lid-1.STL')
    part_b = trimesh.load('STLs/Test Assembly - Pen-1.STL')

    # Get the solid bounding boxes
    bbox_a = part_a.bounding_box
    bbox_b = part_b.bounding_box

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



    # --- Let's test it on your data! ---

    # Test extraction along the X-axis (Index 0)
    x_overlap = check_2d_aabb_overlap(part_a_aux.bounds, part_b_aux.bounds, extraction_axis="x")
    print(f"Do the 2D shadows overlap in the X-extraction path? {x_overlap}")
    # Test extraction along the Y-axis (Index 1)
    y_overlap = check_2d_aabb_overlap(part_a_aux.bounds, part_b_aux.bounds, extraction_axis="y")
    print(f"Do the 2D shadows overlap in the Y-extraction path? {y_overlap}")
    # Test extraction along the Z-axis (Index 2)
    z_overlap = check_2d_aabb_overlap(part_a_aux.bounds, part_b_aux.bounds, extraction_axis="z")
    print(f"Do the 2D shadows overlap in the Z-extraction path? {z_overlap}")

    # --- 5. PyVista Visualization ---
    # Let's draw the part and the 3 extraction arrows!

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