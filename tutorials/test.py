map_canvas = project_map_to_image(
    example["gt_vecs_pts_loc"].data,
    example["gt_vecs_label"].data,
    camera_intrinsics[i],
    camera2ego[i],
)


def project_map_to_image(gt_bboxes_3d, gt_labels_3d, intrinsic, extrinsic, image=None):
    z = 0
    canvas = np.zeros((3, 900, 1600, 3), dtype=np.uint8)
    gt_lines_instance = gt_bboxes_3d.instance_list
    for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d):
        pts = torch.Tensor(list(gt_line_instance.coords))
        pts = pts[:, [1, 0]]
        pts[:, 1] = -pts[:, 1]
        dummy_pts = torch.cat([pts, torch.ones((pts.shape[0], 1)) * z], dim=-1)
        # dummy_pts = torch.cat([dummy_pts, torch.ones((pts.shape[0], 1))], dim=-1)
        points_in_cam_cor = torch.matmul(
            extrinsic[:3, :3].T, (dummy_pts.T - extrinsic[:3, 3].reshape(3, -1))
        )
        points_in_cam_cor = points_in_cam_cor[:, points_in_cam_cor[2, :] > 0]
        if points_in_cam_cor.shape[1] > 1:
            points_on_image_cor = intrinsic[:3, :3] @ points_in_cam_cor
            points_on_image_cor = points_on_image_cor / (
                points_on_image_cor[-1, :].reshape(1, -1)
            )
            points_on_image_cor = points_on_image_cor[:2, :].T
            points_on_image_cor = points_on_image_cor.int().numpy()
        else:
            points_on_image_cor = []

        if image is not None:
            for p in points_on_image_cor:
                cv2.circle(image, tuple(p), 4, (255, 0, 0), -1)
            for i in range(len(points_on_image_cor) - 1):
                cv2.line(
                    image,
                    tuple(points_on_image_cor[i]),
                    tuple(points_on_image_cor[i + 1]),
                    (255, 0, 0),
                    4,
                )

        for i in range(len(points_on_image_cor) - 1):
            cv2.line(
                canvas[int(gt_label_3d)],
                tuple(points_on_image_cor[i]),
                tuple(points_on_image_cor[i + 1]),
                (1, 0, 0),
                4,
            )

        if image is not None:
            cv2.imwrite("./project.png", image)
        canvas = canvas[..., 0]
        canvas = np.transpose(canvas, (1, 2, 0))
        canvas = canvas[::4, ::4, :][1:, ...]
        return canvas
