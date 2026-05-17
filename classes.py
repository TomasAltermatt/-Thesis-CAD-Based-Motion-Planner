import numpy as np
import networkx as nx


class PseudoFace:
    def __init__(self, part, face_indices, extraction_axis):
        self.part = part
        self.face_indices = np.array(list(face_indices), dtype=int)
        self.extraction_axis = extraction_axis
        self.facets = []
        self.focus_facets = []

        self.triangles_3d = part.triangles[self.face_indices]

        axis_idx = {"x": 0, "y": 1, "z": 2}
        all_axes = [0, 1, 2]
        all_axes.remove(axis_idx[extraction_axis])
        self.u_axis, self.v_axis = all_axes[0], all_axes[1]
        
        # Calculate 2D coordinates by projecting 3D triangles on the extraction plane
        self.triangles_2d = self.triangles_3d[:, :, [self.u_axis, self.v_axis]]

        # Calculate the bounding box of the pseudo-face in 2D
        self.u_min = self.triangles_2d[:, :, 0].min()
        self.u_max = self.triangles_2d[:, :, 0].max()
        self.v_min = self.triangles_2d[:, :, 1].min()
        self.v_max = self.triangles_2d[:, :, 1].max()
    
