import open3d
import open3d.visualization.gui as gui_o3d
import numpy as np
import sys
import os
if __name__ == "__main__":
    app = gui_o3d.Application.instance
    app.initialize()
    vis = open3d.visualization.O3DVisualizer()
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(np.random.random((1000, 3)))
    print(type(pts))
    vis.add_geometry(pts)
    vis.show(True)