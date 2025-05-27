import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
import os

def toGLCamera() -> np.ndarray:
    ToGLCamera = np.array([
        [1,  0,  0,  0],
        [0,  -1,  0,  0],
        [0,  0,  -1,  0],
        [0,  0,  0,  1]
    ])
    return ToGLCamera

def fromGLCamera() -> np.ndarray:
    FromGLCamera = np.linalg.inv(toGLCamera())
    return FromGLCamera

def model_matrix_to_extrinsic_matrix(model_matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(model_matrix @ fromGLCamera())

def create_camera_intrinsic_from_size(width: int=1024, height: int=768, hfov: float=60.0, vfov: float=60.0) -> np.ndarray:
    fx = (width / 2.0)  / np.tan(np.radians(hfov)/2)
    fy = (height / 2.0)  / np.tan(np.radians(vfov)/2)
    fx = fy # not sure why, but it looks like fx should be governed/limited by fy
    return np.array(
        [[fx, 0, width / 2.0],
         [0, fy, height / 2.0],
         [0, 0,  1]])

def take_snapshot(vis, save_path: str, name: str, timestamp: bool = False) -> None:
    image_path = os.path.join(save_path, name)
    if timestamp:
        image_path+=f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    image_path += ".png" 
    vis.export_current_image(image_path)

def save_cam_pose(vis, cam_path: str) -> None:
    model_matrix = np.asarray(vis.scene.camera.get_model_matrix())
    extrinsic = model_matrix_to_extrinsic_matrix(model_matrix)
    width, height = vis.size.width, vis.size.height
    intrinsic = create_camera_intrinsic_from_size(width, height)
    saved_view = dict(extrinsic=extrinsic, intrinsic=intrinsic, width=width, height=height)
    with open(cam_path, 'wb') as pickle_file:
        pickle.dump(saved_view, pickle_file)
    print(f"Saved camera {cam_path}")

def load_cam_pose(vis, cam_path: str) -> None:
    if vis is None:
        print("Launch a visualizer to save")
    if os.path.exists(cam_path):
        with open(cam_path, 'rb') as pickle_file:
            saved_view = pickle.load(pickle_file)
        vis.setup_camera(saved_view['intrinsic'], saved_view['extrinsic'], saved_view['width'], saved_view['height'])
        print(f"Loaded camera {cam_path}")
    else:
        print(f"Not found camera {cam_path}. Setting camera to default pose.")
        vis.reset_camera_to_default()

def compute_obj(points: np.ndarray, obj_colors: np.ndarray) -> o3d.t.geometry.PointCloud:
    obj_pcd = o3d.t.geometry.PointCloud()
    obj_pcd.point.positions = points
    obj_pcd.point.colors = obj_colors
    return obj_pcd

def get_obb(obj_pcd: o3d.t.geometry.PointCloud, obb_color=None):
    obb = obj_pcd.get_axis_aligned_bounding_box()
    if obb_color is not None:
        obb.set_color(obb_color)
    else:
        obb.set_color(obj_pcd.point.colors[0])
    return obb

def get_obj_and_obb(masks, points, pcd_colors, obj_colors):
    obb_obj_list = []
    for j in range(masks.shape[-1]):
        if masks[j].sum() > 0 :
            obj_pcd = compute_obj(points[masks[:,j]], pcd_colors[masks[:,j]])
            obj_pcd, obb = get_obb(obj_pcd, obj_colors[masks[:,j]][0])
            obb_obj_list.append([obb, obj_pcd])
    return obb_obj_list

def get_pcd_colors(obj_ids, cmap):
    mapped_ids = obj_ids.copy()
    mapped_ids[mapped_ids>-1] = mapped_ids[mapped_ids>-1]%cmap.shape[0]
    obj_colors = np.take(cmap, mapped_ids, axis=0)
    obj_colors[mapped_ids==-1] = 0
    return obj_colors

def get_obj_ids_and_masks(obj_ids):
    ids = np.unique(obj_ids)
    if not (ids>=0).any():
        return np.array([]), np.array([])
    while ids[0]<0:
        ids = ids[1:]
    masks = np.repeat(obj_ids[:,None], len(ids), axis=-1) == ids[None,:]
    return np.transpose(masks), ids

def get_cmap():
    colours = colors.ListedColormap(plt.cm.tab20b.colors + plt.cm.tab20c.colors)# , name="tab20_extended")
    return colours(np.arange(40))[:,:3].astype(np.float32) # Obtain RGB colour map

def get_camera_centers_lineset(camera_centers, ls, color=np.array([0, 0, 1])):
    num_nodes = len(camera_centers)

    lines = [[x, x + 1] for x in range(num_nodes - 1)]
    colors = np.tile(color, (len(lines), 1))
    ls.point.positions = o3c.Tensor(camera_centers, o3c.float32)
    ls.line.indices = o3c.Tensor(lines, o3c.int32)
    ls.line.colors = o3c.Tensor(colors, o3c.float32)
    return ls
def get_camera_frame(camset, sensor_size, intrinsic, cam2world, scale=0.2, color = np.array([1,0,0])):
    camera_center_in_world = cam2world[:3, 3]
    focal_length = np.mean(np.diag(intrinsic[:2, :2]))
    sensor_corners_in_cam = np.array([
        [0, 0, focal_length],
        [0, sensor_size[1], focal_length],
        [sensor_size[0], sensor_size[1], focal_length],
        [sensor_size[0], 0, focal_length],
    ])
    sensor_corners_in_cam[:, 0] -= intrinsic[0, 2]
    sensor_corners_in_cam[:, 1] -= intrinsic[1, 2]
    sensor_corners_in_world = np.einsum("ij,bj->bi", cam2world, np.hstack((sensor_corners_in_cam,np.ones((4,1)))))[:, :3]
    virtual_image_corners = (
        scale / focal_length *
        (sensor_corners_in_world - camera_center_in_world[np.newaxis]) +
        camera_center_in_world[np.newaxis])

    up = virtual_image_corners[0] - virtual_image_corners[1]
    camera_line_points = np.vstack((
        camera_center_in_world[:3],
        virtual_image_corners[0],
        virtual_image_corners[1],
        virtual_image_corners[2],
        virtual_image_corners[3],
        virtual_image_corners[0] + 0.1 * up,
        0.5 * (virtual_image_corners[0] +
            virtual_image_corners[3]) + 0.5 * up,
        virtual_image_corners[3] + 0.1 * up
    ))

    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
                [1, 2], [2, 3], [3, 4], [4, 1],
                [5, 6], [6, 7], [7, 5]]
    colors = np.tile(color, (len(lines), 1))

    camset.point.positions = o3c.Tensor(
        camera_line_points, o3c.float32)
    camset.line.indices = o3c.Tensor(
        np.array(lines), o3c.int32)
    camset.line.colors = o3c.Tensor(colors, o3c.float32)
    return camset

def create_elements(points, obj_ids, pcd_colors, lvl, skip_obb=False):
        point_cloud = o3d.t.geometry.PointCloud(o3c.Tensor(points, o3c.float32))
        # Set the points
        max_level = obj_ids.shape[1]
        cmap = get_cmap()
        colors_list = []
        obb_list  = []
        obj_masks = []
        if obj_ids is not None:
            for i in range(max_level):
                obj_colors = get_pcd_colors(obj_ids[...,i], cmap)
                colors_list.append(obj_colors)

                masks, ids = get_obj_ids_and_masks(obj_ids[...,i])
                obj_masks.append(np.transpose(masks))
                if not skip_obb:
                    obb_list.append(get_obj_and_obb(masks, points, pcd_colors, obj_colors))
                    
            point_cloud.point.colors = np.asarray(colors_list[lvl], dtype=np.float32)
        return point_cloud, obj_masks, colors_list, obb_list


def create_widgets_window(vis):

    vis.window = gui.Application.instance.create_window("OVO-SLAM options", 400, 160)
    w = vis.window  # for more concise code
    em = w.theme.font_size

    layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                         0.5 * em))

    # Create query inpute widget
    vis._query_in = gui.TextEdit()
    vis._query_in.set_on_value_changed(vis._on_query_value_changed)
    query_button = gui.Button("Update query")
    query_button.set_on_clicked(vis._on_clicked_query)
    query_button.vertical_padding_em = 0
    query_layout = gui.Horiz()
    query_layout.add_child(gui.Label("Query: "))
    query_layout.add_child(vis._query_in)
    query_layout.add_fixed(0.25 * em)
    query_layout.add_child(query_button)
    layout.add_child(query_layout)

    # Slider to change similarity 
    query_th_in = gui.Slider(gui.Slider.DOUBLE)
    query_th_in.set_limits(0.0, 1.0)
    query_th_in.double_value = vis.th
    query_th_in.set_on_value_changed(vis._on_query_th_value_changed)
    updatebutton = gui.Button("Update th")
    updatebutton.set_on_clicked(vis._on_update_querymap_button)
    updatebutton.vertical_padding_em = 0
    query_th_layout = gui.Horiz()
    query_th_layout.add_child(gui.Label("Similarity th: "))
    query_th_layout.add_child(query_th_in)
    query_th_layout.add_fixed(0.25 * em)
    query_th_layout.add_child(updatebutton)  
    layout.add_child(query_th_layout)  

    # Create a checkbox.Check RGB or Instance
    cb = gui.Checkbox("Show object instances")
    cb.set_on_checked(vis._on_cb_pcd_colors)  # set the callback function
    cb.checked = True
    vis.pcd_color_state = "instance"#"image"
    layout.add_child(cb)

    # Check hide ceiling
    cb_ceilling = gui.Checkbox("Hide ceilling")
    cb_ceilling.set_on_checked(vis._on_cb_ceilling)  # set the callback function
    cb_ceilling.checked = True
    layout.add_child(cb_ceilling)

    # Button resume stream
    resumebutton = gui.Button("Resume stream")
    resumebutton.horizontal_padding_em = 0.5
    resumebutton.vertical_padding_em = 0
    resumebutton.set_on_clicked(vis._on_resume_button)
    layout.add_child(resumebutton)

    w.add_child(layout)