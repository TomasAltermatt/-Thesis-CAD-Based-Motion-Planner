import trimesh
import pyvista as pv
# ----------------------------------------------------- MAIN FUNCTIONS ----------------------------------------------
## AABB overlap test functions
def check_2d_aabb_overlap(bounds_a, bounds_b, extraction_axis):
    """
    Squashes 3D bounding boxes onto a 2D plane based on the extraction axis
    and checks if the 2D rectangles overlap.
    extraction_axis: 0 for X, 1 for Y, 2 for Z
    """
    # 1. Figure out which two axes form our 2D "shadow" plane
    # If we extract in Z (2), our 2D plane uses X (0) and Y (1).
    all_axes = [0, 1, 2]
    all_axes.remove(extraction_axis)
    u_axis = all_axes[0]
    v_axis = all_axes[1]
    
    # 2. Extract the Min and Max for the U axis (e.g., the X axis)
    # bounds[0] is Min, bounds[1] is Max
    a_min_u, a_max_u = bounds_a[0][u_axis], bounds_a[1][u_axis]
    b_min_u, b_max_u = bounds_b[0][u_axis], bounds_b[1][u_axis]
    
    # 3. Extract the Min and Max for the V axis (e.g., the Y axis)
    a_min_v, a_max_v = bounds_a[0][v_axis], bounds_a[1][v_axis]
    b_min_v, b_max_v = bounds_b[0][v_axis], bounds_b[1][v_axis]
    
    # 4. The Overlap Math
    # Two lines overlap if one line's MAX is strictly greater than the other's MIN
    u_overlaps = (a_max_u > b_min_u) and (b_max_u > a_min_u)
    v_overlaps = (a_max_v > b_min_v) and (b_max_v > a_min_v)
    
    # 5. The Result: True if BOTH axes overlap (meaning the 2D rectangles intersect)
    return u_overlaps and v_overlaps

def check_COAABB_overlap(bounds_a, bounds_b):
    ## Need to implement
    return


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