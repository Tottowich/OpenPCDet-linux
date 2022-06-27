import open3d
from sklearn.decomposition import non_negative_factorization
import torch
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


class LiveVisualizer:
    def __init__(self,
                    window_name:str='3D Viewer',
                    window_size:tuple=(1920, 1080),
                    point_size:float=1.0,
                    background_color:np.ndarray=np.array((0, 0, 0)),
                    label_colors:list[list]=box_colormap,
                    draw_origin:bool=False,
                    show_labels:bool=False,
                    class_names:list=None,
                    first_cloud:np.ndarray=None,
                    classes_to_visualize:list=None,
                    max_bboxes:int=None,
                ):
        """
        Args:
            window_name (str): window name
            window_size (tuple): window dimensions, (breath, height)
            point_size (float): point size in the visualizer.
            background_color (np.ndarray): background color of the 3D window.
            label_colors (list): list of colors for the labels.
            draw_origin (bool): whether to draw origin.
            show_labels (bool): whether to show labels.
            class_names (list[str]): class names.
            first_cloud (np.ndarray): first cloud to be drawn, used to initialize the window.
            classes_to_visualize (list[int]): classes to be visualized. If None -> visualize all.
            max_bboxes (int): maximum number of bboxes to be drawn, if None => inf, or as many as provided.
        """
        self.label_colors = label_colors
        self.window_name = window_name
        self.window_size = window_size
        self.point_size = point_size
        self.background_color = background_color
        self.draw_origin = draw_origin
        self.first_cloud = first_cloud
        self.class_names = class_names
        self.max_bboxes = max_bboxes if max_bboxes is not None else 1000000 # Not truly infinite but almost :)
        self.show_labels = show_labels
        
        self.lidar_points = open3d.geometry.PointCloud()
        self.vis = open3d.visualization.Visualizer()
        if classes_to_visualize is not None:
            self.classes_to_visualize = [class_names[x] for x in classes_to_visualize]
            print(f"Classes to visualize: {self.classes_to_visualize}")
        else:
            self.classes_to_visualize = class_names
        self.initialize_visual()
        self.pred_boxes = None
        self.pred_boxes = self.initialize_bboxes()
        self.previous_num_bboxes = 0
        self.started = False

    def initialize_visual(self):
        """
        Initialize the visualizer to display the live of point clouds with bounding boxes.
        """
        if self.first_cloud is None:
            # Creates a random point cloud
            self.first_cloud = np.random.random((10000, 3))
        if isinstance(self.first_cloud, torch.Tensor):
            self.first_cloud = self.first_cloud.cpu().numpy()
        # Generate the window and assign
        self.vis.create_window(self.window_name, width=self.window_size[0], height=self.window_size[1])
        self.vis.get_render_option().point_size = self.point_size
        self.vis.get_render_option().background_color = self.background_color
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
        
    def update(self,
               points, 
               ref_boxes=None, 
               ref_labels=None, 
               ref_scores=None):
        """
        Update the visualizer with new points and bounding boxes.
        Args:
            points (np.ndarray): points to be visualized.
            ref_boxes (np.ndarray): reference bounding boxes.
            ref_labels (np.ndarray): reference labels.
            ref_scores (np.ndarray): reference scores.
            point_colors (np.ndarray): point colors.
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(ref_boxes, torch.Tensor):
            ref_boxes = ref_boxes.cpu().numpy()
        if isinstance(ref_labels, torch.Tensor):
            ref_labels = ref_labels.cpu().numpy()
        # Update enviroment points
        self.lidar_points.points = open3d.utility.Vector3dVector(points[:, :3])
        self.lidar_points.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        self.vis.update_geometry(self.lidar_points)
        
        # Update predicted Boxes
        if ref_boxes is not None:
            self.update_bboxes(ref_boxes, scores=ref_scores,labels=ref_labels)
        if not self.started:
            self.vis.run()
        self.vis.poll_events()
        self.vis.update_renderer()
    def update_bboxes(self, bboxes, scores, labels):
        """
        Update the bounding boxes.
        Args:
            bboxes (np.ndarray): bounding boxes.
            scores (np.ndarray): scores.
            labels (np.ndarray): labels.
        """
        if self.pred_boxes is not None:
            for i in range(self.max_bboxes):
                if i < bboxes.shape[0]:
                    if self.class_names[labels[i]-1] in self.classes_to_visualize:
                        axis_angles = np.array([0, 0, bboxes[i][6] + 1e-10])

                        self.pred_boxes[i].R = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
                        self.pred_boxes[i].center = bboxes[i][:3]
                        self.pred_boxes[i].extent = bboxes[i][3:6]
                        self.pred_boxes[i].color = self.label_colors[labels[i]%4]  
                        
                        self.vis.update_geometry(self.pred_boxes[i])
                        self.vis.poll_events()
                        self.vis.update_renderer()
                elif i < self.previous_num_bboxes:
                    #self.shown_bboxes[i] = self.zero_bounding_box()
                    #print(f"Hiding Box {i}")
                    self.pred_boxes[i].center = [0,0,0]
                    self.pred_boxes[i].extent = [0,0,0]
                    self.pred_boxes[i].color = [0,0,0]
                    self.vis.update_geometry(self.pred_boxes[i])
                    self.vis.poll_events()
                    self.vis.update_renderer()
                else:
                    self.previous_num_bboxes = len(bboxes)
                    break
    def zero_bounding_box(self,color=[0, 0, 0]):
        """
        A bounding box with zero-values.
        Used to more effectivly update visualizer.
        """
        axis_angles = np.array([0, 0,  1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(np.array([0,0,0]), rot,np.array([0,0,0]))
        box3d.color = [0.0,0.0,0.0]
        #line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        #line_set = line_set.paint_uniform_color([1,0,0])
        return box3d
    def initialize_bboxes(self):
        """
        Initializes bboxes to make it faster to update (does not continously add geometry, only updates).
        """
        self.pred_boxes = []
        for i in range(self.max_bboxes):
            box3d = self.zero_bounding_box()
            self.pred_boxes.append(box3d)
            self.vis.add_geometry(self.pred_boxes[i])
            self.vis.poll_events()
            self.vis.update_renderer()
        return self.pred_boxes