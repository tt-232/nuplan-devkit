save_data = {}
# basic infos
save_data["metas"] = data_template["metas"]
save_data["img"] = torch.zeros(6, 3, 1, 1)
# ------------ bboxes ------------ #
if len(bbox_list) != 0:
    gt_bboxes_3d = torch.tensor(bbox_list)
    save_data["gt_bboxes_3d"] = gt_bboxes_3d
    save_data["gt_labels_3d"] = torch.tensor(label_list)
else:
    gt_bboxes_3d = torch.empty(0, 9)
    save_data["gt_bboxes_3d"] = gt_bboxes_3d
    save_data["gt_labels_3d"] = torch.empty(0)
# ------------ HDMap ------------ #
anns_results = singapore_map.gen_vectorized_samples(
    MAP_NAME, [ego_x, ego_y], np.deg2rad(ego_yaw_deg - 90)
)
gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
    gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
else:
    gt_vecs_pts_loc = to_tensor(anns_results["gt_vecs_pts_loc"])
    try:
        gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
    except:
        gt_vecs_pts_loc = gt_vecs_pts_loc
bev_map = visualize_bev_hdmap(
    gt_vecs_pts_loc, gt_vecs_label, [200, 200], vis_format="polyline_pts"
)
bev_map = bev_map.transpose(2, 0, 1)  # N_channel, H, W
save_data["bev_hdmap"] = to_tensor(bev_map.copy())
# ----------------bev HDmap to img------------------#
lidar2camera = data_template["lidar2camera"]
camera2ego = data_template["camera2ego"]
camera_intrinsics = data_template["camera_intrinsics"]
save_data["metas"]["location"] = "singapore-onenorth"
save_data["metas"]["description"] = "Peds, construction zone"
save_data["metas"]["ego_pos"] = torch.Tensor(
    [
        [np.cos(ego_yaw), -np.sin(ego_yaw), 0, ego_x],
        [np.sin(ego_yaw), np.cos(ego_yaw), 0, ego_y],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
layout_canvas = []
for i in range(len(lidar2camera)):
    # image = cv2.imread(img_file_list[i])
    lidar2image = camera_intrinsics[i] @ lidar2camera[i].T
    import pdb

    pdb.set_trace()
    map_canvas = project_map_to_image(
        gt_vecs_pts_loc, gt_vecs_label, camera_intrinsics[i], camera2ego[i]
    )
    gt_bboxes = LiDARInstance3DBoxes(
        save_data["gt_bboxes_3d"], box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
    ).convert_to(Box3DMode.LIDAR)
    box_canvas = project_box_to_image(
        gt_bboxes, save_data["gt_labels_3d"], lidar2image, object_classes=object_classes
    )
    layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))
layout_canvas = np.stack(layout_canvas, axis=0)
layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))  # 6, N_channel, H, W
save_data["layout_canvas"] = to_tensor(layout_canvas)
# ---------------ref pose------------------#
save_data["relative_pose"] = torch.matmul(
    torch.inverse(save_data["metas"]["ego_pos"]), last_pose
)
