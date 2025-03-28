import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from typing import Any, Dict, List
from pathlib import Path
import numpy as np
import threading
import time
import os

from ..utils import vis_utils

class Visualizer:
    def __init__(self, semantic_module = None, scene_name: str = "", name: str ="", save_path: str | None = None):
        self.is_done = False
        self.semantic_module = semantic_module
        if semantic_module is not None:
            self.init_semantic_module()
        self.cmap = plt.get_cmap('viridis')
        self.last_query = ""
        self.query_context = [] #["object", "stuff", "wall", "floor", "thing", "table"]

        self.scene_name = scene_name
        #self.save_cam_path =  f"./snapshots/{scene_name}"
        self.save_path = Path(f"./snapshots/{scene_name}")
        if save_path is not None:
            if isinstance(save_path, str):
                self.save_path = Path(f"{save_path}/snapshots/{scene_name}")
            else:
                self.save_path = save_path/f"snapshots/{scene_name}"
            os.makedirs(self.save_path, exist_ok=True)
            #self.save_cam_path = self.save_path
        else:
            os.makedirs(self.save_cam_path, exist_ok=True)
        self.save_cam_path = save_path

        self.name = name
        self.obj_cmap = vis_utils.get_cmap()
        self.th = 0.5
        self.cam_counter = 0
        self.n_snap=0
        self.timestamp = False
        self.skip_obb = True
        self.state = ""
        self.pcd_color_state = "image"
        self.stream_event = threading.Event()
        self.stream_lock =  threading.Lock()
        self.heatmap_lock = threading.Lock()
        self.pcd_color_lock = threading.Lock()
        self.pcd_lock = threading.Lock()
        self.done_lock = threading.Lock()

        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()
        self.main_vis = o3d.visualization.O3DVisualizer(self.name, 1280,720)
        self.main_vis.add_action("Take snapshot",
                                 self._take_snapshot)
        self.main_vis.add_action("Save camera pose",
                                 self.save_camera_pose)
        self.main_vis.add_action("Load camera pose",
                                 self.load_camera_pose)
        self.main_vis.set_on_close(self.on_main_window_closing)
        self.main_vis.show_skybox(False)
        self.main_vis.enable_raw_mode(True)

    def streaming_decorator(func):
        """A decorator that temporaly pauses streaming if its running.
        Args:
            func: The function to be decorated.
        Returns:
            The wrapper function.
        """
        def wrapper(self, *args, **kwargs):
            if self.stream_event.is_set():
                with self.stream_lock:
                    self.stream_event.clear()
                    func(self, *args, **kwargs)
                    self.stream_event.set()
            else:
                func(self, *args, **kwargs)
            return
        return wrapper
    
    def init_semantic_module(self):
        self.ids = list(self.semantic_module.objects.keys())
        self.n_objs = len(self.ids)

    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close    

    def _get_snapshot_name(self):
        if self.state == "query":
            name = f"query_{self.scene_name}_{self.last_query}_{self.th:.3f}_{self.n_snap}"
        elif self.state == "streaming":
            name = f"stream_{self.scene_name}_{self.n_snap}"
        else:
            name = f"snap_{self.scene_name}_{self.n_snap}"
        return name
    
    def _take_snapshot(self, vis):
        vis_utils.take_snapshot(vis, str(self.save_path), self._get_snapshot_name(), timestamp=self.timestamp)
        self.n_snap +=1

    def take_snapshot(self):
        return self._take_snapshot(self.main_vis)

    def save_camera_pose(self, vis):
        cam_path = self.save_cam_path / f"cam_{self.cam_counter}.pickle"
        vis_utils.save_cam_pose(vis, cam_path)

    def load_camera_pose(self, vis):
        cam_path = self.save_cam_path / f"cam_{self.cam_counter}.pickle"
        vis_utils.load_cam_pose(vis, cam_path)
    
    def _on_update_querymap_button(self):
        with self.stream_lock:
            self.stream_event.clear()
            return self._update_query_vis()

    def _on_query_th_value_changed(self, new_value):
        self.th = new_value

    def _on_query_value_changed(self, query):
        self.last_query = query
        if self.state == "streaming":
            return self._query_stream_process()
        else:
            return self._query_locally()
    
    def _on_clicked_query(self):
        self.last_query = self._query_in.text_value
        if self.state == "streaming":
            return self._query_stream_process()
        else:
            return self._query_locally()

    def _query_locally(self):
        with self.heatmap_lock:
            self.last_query_map  = self.semantic_module.query([self.last_query]+self.query_context).cpu().numpy()
        self._update_query_vis()
    
    def _query_stream_process(self):
        with self.stream_lock:
            self.stream_event.clear()
            # Send signal to slam process
            with self.query_flag.get_lock():
                self.query_pipe.send([self.last_query]+self.query_context)
                self.query_flag.value = 1

            # Wait for response
            while self.query_flag.value !=2:
                time.sleep(2)
            # Read query_map
            with self.query_flag.get_lock() and self.heatmap_lock:
                self.last_query_map = self.query_pipe.recv()
                self.query_flag.value = 0
            # Update vis
            return self._update_query_vis()

    #@streaming_decorator
    def _update_query_vis(self):
        with self.heatmap_lock:
            heat_map = self.last_query_map.copy()
        sim_mask = (heat_map > self.th).squeeze()
        obj_colors = self.cmap(heat_map[sim_mask])[...,:3]
        n_objs = self.obj_masks.shape[0]
        sim_idxs = np.arange(n_objs)[sim_mask[:n_objs]]

        with self.pcd_color_lock:
            new_pcd_colors = self.pcd_colors.copy()
            for i, idx in enumerate(sim_idxs):
                new_pcd_colors[self.obj_masks[idx]] = obj_colors[i] 

            self.cloud.point.colors = new_pcd_colors        
            self.main_vis.update_geometry("pcd",self.cloud , 20)
        self.main_vis.post_redraw()
        return False

    @streaming_decorator
    def _on_cb_pcd_colors(self, is_checked):
        with self.pcd_color_lock:
            if is_checked:
                self.pcd_color_state = "instance"
                self.cloud.point.colors = self.obj_colors
            else:
                self.pcd_color_state = "image"
                self.cloud.point.colors = self.pcd_colors
            self.main_vis.update_geometry("pcd", self.cloud, 20)

    @streaming_decorator
    def _on_cb_ceilling(self, is_checked):
        if is_checked:
            self.mask_ceiling = True
        else:
            self.mask_ceiling = False

    def _on_resume_button(self):
        with self.stream_lock:
            self.stream_event.set()

    def _on_reset_vis_button(self):
        with self.pcd_color_lock:
            if self.pcd_color_state == "instance":
                
                self.cloud.point.colors = self.obj_colors
            elif self.pcd_color_state == "image":
                self.cloud.point.colors = self.pcd_colors

            self.main_vis.update_geometry("pcd", self.cloud, 20)

    def visualize_and_query(self, points, obj_ids, pcd_colors):
        self.state = "query"
       
        #self.main_vis.show_settings = False 
        self.main_vis.scene_shader = o3d.visualization.O3DVisualizer.UNLIT

        self.mask_ceiling = True
        self.app.add_window(self.main_vis)
        self.create_widgets_window()

        self.cloud = o3d.t.geometry.PointCloud(o3c.Tensor(np.asarray(points), o3c.float32))
        self.obj_colors = vis_utils.get_pcd_colors(obj_ids, self.obj_cmap)
        self.pcd_colors = np.asarray(pcd_colors/255.) 
        self.obj_masks, ids = vis_utils.get_obj_ids_and_masks(obj_ids)
        self.n_objs = len(ids)
        with self.pcd_color_lock:
            if self.pcd_color_state == "image":
                self.cloud.point.colors = self.pcd_colors
            else:
                self.cloud.point.colors = self.obj_colors
        
        threading.Thread(target=self._thread_query).start()
        self.app.run()

    def _thread_query(self):
        bounds = self.cloud.get_axis_aligned_bounding_box()

        def add_first_cloud():
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            self.main_vis.setup_camera(60, bounds.get_center().numpy(),
                                       bounds.get_center().numpy()+[0., 0., -3.],[0., -1., 0.])
            self.main_vis.add_geometry("pcd", self.cloud, mat)
            self.main_vis.reset_camera_to_default()

        o3d.visualization.gui.Application.instance.post_to_main_thread(self.main_vis, add_first_cloud)
        
        while True:
            with self.done_lock:
                if self.is_done:
                    break

            time.sleep(0.1)

    def create_widgets_window(self):
        self.window = gui.Application.instance.create_window("Query options", 400, 160)
        w = self.window  
        em = w.theme.font_size

        layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                         0.5 * em))

        # Create query inpute widget
        self._query_in = gui.TextEdit()
        self._query_in.set_on_value_changed(self._on_query_value_changed)
        query_button = gui.Button("Update query")
        query_button.set_on_clicked(self._on_clicked_query)
        query_button.vertical_padding_em = 0
        query_layout = gui.Horiz()
        query_layout.add_child(gui.Label("Query: "))
        query_layout.add_child(self._query_in)
        query_layout.add_fixed(0.25 * em)
        query_layout.add_child(query_button)
        layout.add_child(query_layout)

        # Slider to change similarity 
        query_th_in = gui.Slider(gui.Slider.DOUBLE)
        query_th_in.set_limits(0.0, 1.0)
        query_th_in.double_value = self.th
        query_th_in.set_on_value_changed(self._on_query_th_value_changed)
        updatebutton = gui.Button("Update th")
        updatebutton.set_on_clicked(self._on_update_querymap_button)
        updatebutton.vertical_padding_em = 0
        query_th_layout = gui.Horiz()
        query_th_layout.add_child(gui.Label("Similarity th: "))
        query_th_layout.add_child(query_th_in)
        query_th_layout.add_fixed(0.25 * em)
        query_th_layout.add_child(updatebutton)  
        layout.add_child(query_th_layout)  

        # Create a checkbox.Check RGB or Instance
        cb = gui.Checkbox("Show object instances")
        cb.set_on_checked(self._on_cb_pcd_colors)  # set the callback function
        cb.checked = False
        self.pcd_color_state = "instance" if cb.checked else "image"
        layout.add_child(cb)

        # Check hide ceiling
        cb_ceilling = gui.Checkbox("Hide ceilling")
        cb_ceilling.set_on_checked(self._on_cb_ceilling)  # set the callback function
        cb_ceilling.checked = True
        layout.add_child(cb_ceilling)
        if self.state == "streaming":
            # Button resume stream
            resumebutton = gui.Button("Resume stream")
            resumebutton.horizontal_padding_em = 0.5
            resumebutton.vertical_padding_em = 0
            resumebutton.set_on_clicked(self._on_resume_button)
            layout.add_child(resumebutton)
        elif self.state == "query":
            # Button resume stream
            resumebutton = gui.Button("Reset vis")
            resumebutton.horizontal_padding_em = 0.5
            resumebutton.vertical_padding_em = 0
            resumebutton.set_on_clicked(self._on_reset_vis_button)
            layout.add_child(resumebutton)

        w.add_child(layout)

    def stream_and_query(self,  mpqueue, cam_intrinsic, query_data, show=True):
        self.state = "streaming"

        self.main_vis.show_settings =  False
        self.main_vis.scene_shader = o3d.visualization.O3DVisualizer.UNLIT

        self.mpqueue = mpqueue
        self.cam_intrinsic = cam_intrinsic
        self.mask_ceiling = True
        self.query_flag = query_data[0]
        self.query_pipe = query_data[1]
        self.app.add_window(self.main_vis)
        self.create_widgets_window()
        threading.Thread(target=self._stream, args=(mpqueue, cam_intrinsic, show)).start()
        self.stream_event.set()
        self.app.run()
            
    def _stream(self, mpqueue, cam_intrinsic, show):

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat_cam = o3d.visualization.rendering.MaterialRecord()
        mat_cam.shader = "unlitLine"

        cam_lineset = o3d.t.geometry.LineSet()
        step = 0
        while True:
            with self.done_lock:
                if self.is_done:
                    break

            if self.stream_event.wait():
                with self.stream_lock:
                    points, obj_ids, colors, c2w = mpqueue.get()
                    points, colors, c2w = points.astype(np.float32), colors.astype(np.float32), c2w.astype(np.float32)
                    vis_utils.get_camera_frame(cam_lineset, [cam_intrinsic["width"], cam_intrinsic["height"]], cam_intrinsic["intrinsic"], c2w, scale = 0.1)

                    if self.mask_ceiling:
                        points_w = points
                        celing_z = points_w[:,-1].max()-0.2
                        mask = points_w[:,-1]<celing_z
                        points, obj_ids, colors = points_w[mask,:], obj_ids[mask], colors[mask,:]

                    self.cloud = o3d.t.geometry.PointCloud(o3c.Tensor(points, o3c.float32))
                    self.obj_colors = vis_utils.get_pcd_colors(obj_ids, self.obj_cmap)

                    self.pcd_colors = colors/255.
                    self.obj_masks, ids = vis_utils.get_obj_ids_and_masks(obj_ids)
                    self.n_objs = len(ids)
                    
                    with self.pcd_color_lock:
                        if self.pcd_color_state == "image":
                            self.cloud.point.colors = self.pcd_colors
                        else:
                            self.cloud.point.colors = self.obj_colors

                    self.main_vis.remove_geometry("pcd")
                    self.main_vis.add_geometry("pcd", self.cloud, mat)
                        
                    self.main_vis.remove_geometry(f"cam_{step-1}")
                    self.main_vis.add_geometry(f"cam_{step}", cam_lineset, mat_cam)
                    if step == 0:
                        self.load_camera_pose(self.main_vis)
                    step+=1
                    self.main_vis.post_redraw()

                    o3d.visualization.gui.Application.instance.post_to_main_thread(self.main_vis, self.take_snapshot)
                    if self.main_vis.is_visible and not show:
                        self.window.show(False)
                        self.main_vis.show(False)


def stream_pcd(semantic_module, mpqueue, query_data, cam_intrinsic, scene_name, output_path, show):
    with open("./temp.txt", "w") as f:
        vis = Visualizer(semantic_module, scene_name, "Stream", save_path=output_path) 
        vis.stream_and_query(mpqueue, cam_intrinsic, query_data, show)

def visualize_gt_vs_pred(points, gt, pred, labels, labels_idx):
    # Create an Open3D PointCloud object
    IDX = 0
    base_color = None
    n_labels = labels.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    scene_elements = [pcd]

    def next_obj(vis):
        nonlocal IDX, pcd, gt, pred, labels, labels_idx
        IDX =(IDX+1)%(n_labels)
        color = np.zeros_like(points)
        mask_gt = gt==labels_idx[IDX]
        mask_pre = pred==labels_idx[IDX]
        color[np.logical_and(mask_gt, mask_pre)] = np.array([0,255,0]) 
        color[np.logical_and(mask_gt, ~mask_pre)] = np.array([0,0,255]) 
        color[np.logical_and(~mask_gt, mask_pre)] = np.array([255,0,0]) 
        pcd.colors = o3d.utility.Vector3dVector(color)
        vis.update_geometry(pcd)
        vis.update_renderer()
        print(f"Label {labels[IDX]}")
        return True    
    
    def prev_obj(vis):
        nonlocal IDX, pcd, gt, pred, labels, labels_idx, base_color

        IDX =(IDX-1) if IDX >0 else n_labels-1

        color = np.zeros_like(points)
        mask_gt = gt==labels_idx[IDX]
        mask_pre = pred==labels_idx[IDX]
        color[np.logical_and(mask_gt, mask_pre)] = np.array([0,255,0]) 
        color[np.logical_and(mask_gt, ~mask_pre)] = np.array([0,0,255]) 
        color[np.logical_and(~mask_gt, mask_pre)] = np.array([255,0,0]) 
        pcd.colors = o3d.utility.Vector3dVector(color)
        vis.update_geometry(pcd)
        vis.update_renderer()
        print(f"Label {labels[IDX]}")
        return True        
    
    key_to_callback = {}
    key_to_callback[ord("D")] = next_obj
    key_to_callback[ord("A")] = prev_obj
    o3d.visualization.draw_geometries_with_key_callbacks(scene_elements, key_to_callback)
    return

def idxToRGB(seg_image, rgb_image = None, alpha=0.6,max_idx=40):
    """
    Maps each pixel in an image with idx values to RGB. Assume idxs<100 

    Args:
        seg_image (np.array): The 2D image with idx values for each pixel.
        map_data (list): The list of color information from the CSV file.

    Returns:
        PIL.Image: The image with pixels converted to RGB colors.
    """
    #assert seg_image.max()<max_idx, f"Error, more than {max_idx} segments"

    colours = colors.ListedColormap(plt.cm.tab20b.colors + plt.cm.tab20c.colors)# , name="tab20_extended")
    cmap = colours(np.arange(max_idx))[:,:3] # Obtain RGB colour map
    # Create a new RGB image
    if rgb_image is None:
        rgb_image = np.zeros(list(seg_image.shape)+[3])
        alpha = 1.0
    idxs_image = seg_image[...,None]
    legend_classes = {"labels":[], "rgb":[]}
    # Map each pixel value to its color
    for i in range(idxs_image.max()+1):
        mask = (idxs_image==i).squeeze(-1)
        if mask.sum()>0:
            rgb_image[mask] = cmap[i%len(cmap)]*alpha +(1-alpha)*rgb_image[mask]
            legend_classes["labels"].append(str(i))
            legend_classes["rgb"].append(cmap[i%len(cmap)])
    rgb_image[idxs_image[...,0]==-1,:] = 0
    return rgb_image, legend_classes


def visualize_3d_points_obj_id_and_obb(points, obj_ids, colors):
    global LVL, BBOX_IDX, BBOX_STATE, PCD_STATE
    LVL = 0
    BBOX_IDX = 0
    BBOX_STATE = False
    PCD_STATE = True
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    # Set the points
    point_cloud.points = o3d.utility.Vector3dVector(points)
    scene_elements = [point_cloud]
    max_level = obj_ids.shape[1]
    if obj_ids is not None:
        colors_list = []
        obb_list  = []
        for i in range(max_level):
            colors_lvl = idxToRGB(obj_ids[...,i])[0]
            colors_list.append(colors_lvl)
            obb_lvl_list = []
            for j in range(-1,  max(obj_ids[:,i]+1)):
                mask = obj_ids[...,i] == j
                if mask.sum() > 0 :
                    obj_pcd = o3d.geometry.PointCloud()
                    obj_pcd.points = o3d.utility.Vector3dVector(points[mask])
                    obj_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
                    obb = obj_pcd.get_axis_aligned_bounding_box()
                    obb.color = colors_lvl[mask][0]
                    obb_lvl_list.append([obb, obj_pcd])
            obb_list.append(obb_lvl_list)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_list[LVL])
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False
    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.axis('off')
        plt.show()
        return False
    def bboxes(vis):
        global BBOX_STATE
        for obb in obb_list[LVL]:
            if BBOX_STATE:
                vis.add_geometry(obb[0], reset_bounding_box=False)
            else:
                try:
                    vis.remove_geometry(obb[0], reset_bounding_box=False)
                except:
                    vis.clear_geometries()
        vis.update_renderer()
        BBOX_STATE = not BBOX_STATE
        return True
    
    def next_obj(vis):
        global BBOX_IDX
        BBOX_IDX =(BBOX_IDX+1)%(len(obb_list[LVL])+1)

        if BBOX_IDX < len(obb_list[LVL]):
            vis.add_geometry(obb_list[LVL][BBOX_IDX][1], reset_bounding_box=False)
        if BBOX_IDX > 0:
            try:
                vis.remove_geometry(obb_list[LVL][BBOX_IDX-1][1], reset_bounding_box=False)
            except:
                vis.clear_geometries()
        vis.update_renderer()
        return True    
    
    def prev_obj(vis):
        global BBOX_IDX

        if BBOX_IDX < len(obb_list[LVL]):
            vis.add_geometry(obb_list[LVL][BBOX_IDX-1][1], reset_bounding_box=False)
        if BBOX_IDX > 0:
            try:
                vis.remove_geometry(obb_list[LVL][BBOX_IDX][1], reset_bounding_box=False)
            except:
                vis.clear_geometries()
        BBOX_IDX =(BBOX_IDX-1) if BBOX_IDX >0 else len(obb_list[LVL])
        vis.update_renderer()
        return True        

    def full_pcd(vis):
        global PCD_STATE
        if PCD_STATE:
            try:
                vis.remove_geometry(point_cloud, reset_bounding_box=False)
            except:
                vis.clear_geometries()
        else:
            vis.add_geometry(point_cloud, reset_bounding_box=False)
        PCD_STATE = not PCD_STATE
        vis.update_renderer()
        return True
    
    def change_level(vis):
        global LVL
        LVL = (LVL+1) % max_level 
        vis.clear_geometries()
        point_cloud.colors =  o3d.utility.Vector3dVector(colors_list[LVL])
        vis.add_geometry(point_cloud, reset_bounding_box=False)
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    key_to_callback[ord("F")] = full_pcd
    key_to_callback[ord("B")] = bboxes
    key_to_callback[ord("L")] = change_level
    key_to_callback[ord("D")] = next_obj
    key_to_callback[ord("A")] = prev_obj
    o3d.visualization.draw_geometries_with_key_callbacks(scene_elements, key_to_callback)