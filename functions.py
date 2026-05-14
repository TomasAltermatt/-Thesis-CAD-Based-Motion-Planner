import trimesh
import pyvista as pv
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



def check_COAABB_overlap(a_lims, b_lims, epsilon = 0.05):
    a_min_u, a_max_u = a_lims[0]
    a_min_v, a_max_v = a_lims[1]
    b_min_u, b_max_u = b_lims[0]
    b_min_v, b_max_v = b_lims[1]

    # Define lu and lv
    lu = (b_max_u - b_min_u) if (b_max_u - b_min_u) <= (a_max_u - a_min_u) else (a_max_u - a_min_u)
    lv = (b_max_v - b_min_v) if (b_max_v - b_min_v) <= (a_max_v - a_min_v) else (a_max_v - a_min_v)

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

    # 1. Load your individual facet models
    # Replace these strings with the actual names of your STL files
    part_a = trimesh.load('STLs/Test Assembly - Lid-1.STL')
    part_b = trimesh.load('STLs/Test Assembly - Pen-1.STL')

    #print(dir(part_a))

    # 2. Get the solid bounding boxes
    bbox_a = part_a.bounding_box
    bbox_b = part_b.bounding_box

    # 3. Wrap the trimesh objects so PyVista can read them
    pv_a = pv.wrap(part_a)
    pv_b = pv.wrap(part_b)
    pv_bbox_a = pv.wrap(bbox_a)
    pv_bbox_b = pv.wrap(bbox_b)

    # 4. Set up the modern PyVista Plotter
    plotter = pv.Plotter()

    # Add the solid parts (Let's make the Lid slightly transparent too!)
    plotter.add_mesh(pv_a, color='lightgray', opacity=0.6)
    plotter.add_mesh(pv_b, color='blue', opacity=1.0)

    # Add the Bounding Boxes strictly as WIREFRAMES
    # This guarantees they will never hide the parts inside them
    plotter.add_mesh(pv_bbox_a, style='wireframe', color='red', line_width=2)
    plotter.add_mesh(pv_bbox_b, style='wireframe', color='green', line_width=3)

    # Show the interactive window
    plotter.show()


    # --- Let's test it on your data! ---

    # Test extraction along the Z-axis (Index 2)
    z_overlap = check_2d_aabb_overlap(part_a.bounds, part_b.bounds, extraction_axis=2)
    print(f"Do the 2D shadows overlap in the Z-extraction path? {z_overlap}")

    # Test extraction along the X-axis (Index 0)
    x_overlap = check_2d_aabb_overlap(part_a.bounds, part_b.bounds, extraction_axis=0)
    print(f"Do the 2D shadows overlap in the X-extraction path? {x_overlap}")