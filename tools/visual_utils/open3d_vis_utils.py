"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import open3d.visualization.gui as gui_o3d
from sklearn.decomposition import non_negative_factorization
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()
def initialize_bboxes(vis,num):
    bboxes = []
    for i in range(num):
        axis_angles = np.array([0, 0,  1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(np.array([0,0,0]), 0,np.array([0,0,0]))
        line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color([0,0,0])
        bboxes.append(line_set)
        vis.add_geometry(line_set)
    return bboxes

def create_live_scene_GUI(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=False,class_names=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    shown_bboxes = initialize_bboxes(vis,100)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    # Add surronding
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    # Add ground truth Boxes
    if gt_boxes is not None:
        draw_box(vis, gt_boxes, (0, 0, 1),ref_labels=ref_labels,ref_scores=ref_scores,class_names=class_names)
    # Add predicted Boxes
    if ref_boxes is not None:
        draw_box(vis, ref_boxes, (0, 1, 0),ref_labels=ref_labels,ref_scores=ref_scores,class_names=class_names)
    vis.run()
    return vis,pts,len(ref_boxes)
def update_bboxes(ref_bboxes,shown_bboxes):
    pass

def update_live_scene_GUI(vis,pts,shown_bboxes,points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=False,class_names=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    
    #vis.clear_geometries()

    #if draw_origin:
    #    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    #    vis.add_geometry(axis_pcd)
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    vis.update_geometry(pts)
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
    vis.update_renderer()
    vis.poll_events()





def create_live_scene(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=False,class_names=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    vis.reset_view_point(False)
    #mat = open3d.visualization.rendering.Material()
    #mat.shader = "defaultUnlit"
    #mat.point_size = 5 * w.scaling
    #mat.base_color = (0, 0, 0, 1)
    #vis.create_window()
    #vis.point_size = 1.0
    #black = np.ndarra((4,1))
    #vis.set_background(np.zeros((4,1),dtype=np.float32))

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        draw_box(vis, gt_boxes, (0, 0, 1),ref_labels=ref_labels,ref_scores=ref_scores,class_names=class_names)

    if ref_boxes is not None:
        draw_box(vis, ref_boxes, (0, 1, 0),ref_labels=ref_labels,ref_scores=ref_scores,class_names=class_names)
    vis.run()
    return vis,pts
def update_live_scene(vis,pts,points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=False,class_names=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    
    #vis.clear_geometries()

    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    vis.add_geometry(pts)
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
    vis.update_renderer()
    vis.poll_events()

def translate_boxes_to_open3d_instance(gt_boxes):
    """
            4---------6
           /|        /|
          5---------3 |
          | |       | |
          | 7-------|-1
          |/        |/
          2---------0
    """

    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    #print(f"Center: {center}, lwh: {lwh}, axis_angles: {axis_angles}")
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    #lines = np.asarray(line_set.lines)
    #lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    #line_set.lines = open3d.utility.Vector2iVector(lines)
    #print(f"Line set: {np.array(line_set.lines)}")
    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, ref_scores=None,class_names=None):
    
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        #print(f"box3d.shape: {box3d.shape}")
        if ref_labels is None:
            line_set.paint_uniform_color(color)
            #vis.add_3d_label(box3d.get_center()+[0,0,box3d.get_max_bound(2)],f"{ref_scores[i]}: {class_names[ref_labels[i]]}")
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]%4])
            #vis.add_3d_label(box3d.get_center()+[0,0,box3d.get_max_bound(2)], f"{ref_scores[i]}: {class_names[ref_labels[i]]}")
            

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


class LiveVisualizer:
    def __init__(self, 
                    max_bboxes,
                    draw_origin=False,
                    show_labels=False,
                    class_names=None,
                    first_cloud=None,
                ):
        """
        Args:
            max_bboxes (int): maximum number of bboxes to be drawn.
            draw_origin (bool): whether to draw origin.
            show_labels (bool): whether to show labels.
            class_names (list[str]): class names.
        """
        self.first_cloud = first_cloud
        self.max_bboxes = max_bboxes
        self.draw_origin = draw_origin
        self.show_labels = show_labels
        self.class_names = class_names
        self.lidar_points = open3d.geometry.PointCloud()
        self.vis = open3d.visualization.Visualizer()
        self.initialize_visual()
        self.shown_bboxes = None
        self.shown_bboxes = self.initialize_bboxes()
        self.previous_num_bboxes = 0
        self.started = False

    def initialize_visual(self):
        if isinstance(self.first_cloud, torch.Tensor):
            self.first_cloud = self.first_cloud.cpu().numpy()
        #if isinstance(gt_boxes, torch.Tensor):
        #    gt_boxes = gt_boxes.cpu().numpy()
        #if isinstance(ref_boxes, torch.Tensor):
        #    ref_boxes = ref_boxes.cpu().numpy()
        self.vis.create_window()
        self.vis.get_render_option().point_size = 1.0
        self.vis.get_render_option().background_color = np.zeros(3)
        if self.draw_origin:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            self.vis.add_geometry(axis_pcd)
        if self.first_cloud is not None:
            #print(f"First clout: {self.first_cloud[:,:3].shape}")
            #print(f"type self.lidar_points: {type(self.lidar_points)}")
            self.lidar_points.points = open3d.utility.Vector3dVector(self.first_cloud[:, :3])
            self.lidar_points.colors = open3d.utility.Vector3dVector(np.ones((self.first_cloud.shape[0], 3)))
            #pts.colors = open3d.utility.Vector3dVector(np.ones((self.first_cloud.shape[0], 3)))
            self.vis.add_geometry(self.lidar_points)
        #pts = open3d.geometry.PointCloud()
        #pts.points = open3d.utility.Vector3dVector(self.first_cloud[:, :3])    
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def update(self,points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=(1,1,1), draw_origin=False,class_names=None):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(ref_boxes, torch.Tensor):
            ref_boxes = ref_boxes.cpu().numpy()
        if isinstance(ref_labels, torch.Tensor):
            ref_labels = ref_labels.cpu().numpy()
        # Add surronding
        self.lidar_points.points = open3d.utility.Vector3dVector(points[:, :3])
        self.lidar_points.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        
        self.vis.update_geometry(self.lidar_points)
        # Add ground truth Boxes
        #if gt_boxes is not None:
        #    draw_box(gt_boxes, (0, 0, 1),ref_labels=ref_labels,ref_scores=ref_scores,class_names=class_names)
        # Add predicted Boxes
        if ref_boxes is not None:
            self.update_bboxes(ref_boxes, scores=ref_scores,labels=ref_labels,class_names=class_names)
        if not self.started:
            self.vis.run()
        self.vis.poll_events()
        self.vis.update_renderer()
    def update_bboxes(self, bboxes, scores, labels, class_names):
        #print(f"BBoxes: {bboxes.shape}")
        #print(f"Labels: {labels.shape}")
        if self.shown_bboxes is not None:
            for i in range(self.max_bboxes):
                if i < bboxes.shape[0]:
                    #print(f"label {labels[i]}")
                    #print(f"Showing Box {i}")
                    box3d = translate_boxes_to_open3d_instance(bboxes[i])
                    axis_angles = np.array([0, 0, bboxes[i][6] + 1e-10])

                    self.shown_bboxes[i].R = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
                    self.shown_bboxes[i].center = bboxes[i][:3]
                    self.shown_bboxes[i].extent = bboxes[i][3:6]
                    self.shown_bboxes[i].color = box_colormap[labels[i]%4]  
                    
                    #self.shown_bboxes[i].lines = line_set.lines
                    #self.shown_bboxes[i].paint_uniform_color(box_colormap[labels[i]%4])
                    self.vis.update_geometry(self.shown_bboxes[i])
                    self.vis.poll_events()
                    self.vis.update_renderer()
                elif i < self.previous_num_bboxes:
                    #self.shown_bboxes[i] = self.zero_bounding_box()
                    #print(f"Hiding Box {i}")
                    self.shown_bboxes[i].center = [0,0,0]
                    self.shown_bboxes[i].extent = [0,0,0]
                    self.shown_bboxes[i].color = [0,0,0]
                    self.vis.update_geometry(self.shown_bboxes[i])
                    self.vis.poll_events()
                    self.vis.update_renderer()
                else:
                    self.previous_num_bboxes = len(bboxes)
                    break
    def zero_bounding_box(self,color=[0, 0, 0]):
        axis_angles = np.array([0, 0,  1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(np.array([0,0,0]), rot,np.array([0,0,0]))
        box3d.color = [0.0,0.0,0.0]
        #line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        #line_set = line_set.paint_uniform_color([1,0,0])
        return box3d
    def initialize_bboxes(self):
        self.shown_bboxes = []
        for i in range(self.max_bboxes):
            box3d = self.zero_bounding_box()
            self.shown_bboxes.append(box3d)
            self.vis.add_geometry(self.shown_bboxes[i])
            self.vis.poll_events()
            self.vis.update_renderer()
        return self.shown_bboxes
    def _translate_boxes_to_open3d_instance(gt_boxes):
        """
              4---------6
             /|        /|
            5---------3 |
            | |       | |
            | 7-------|-1
            |/        |/
            2---------0
        """
        center = gt_boxes[0:3]
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
        #lines = np.asarray(line_set.lines)
        #lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        #line_set.lines = open3d.utility.Vector2iVector(lines)

        return line_set, box3d