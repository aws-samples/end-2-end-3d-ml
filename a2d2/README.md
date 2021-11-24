For convenience we have supplied pickle files used with the training dataloader in this folder. To create these a2d2 pickle files, run the following code:

```
from mmdetection3d.tools.data_converter import kitti_converter
from a2d2.a2d2_database import create_groundtruth_database

kitti_converter.create_kitti_info_file('A2D2')

create_groundtruth_database(
    dataset_class_name = 'A2D2Dataset',
    data_path          = '/path/to/camera_lidar_semantic_bboxes',
    info_prefix        = 'a2d2',
    info_path          = '/path/to/camera_lidar_semantic_bboxes/a2d2_infos_train.pkl',
    mask_anno_path     = 'instances_train.json',
    lidar_only         = True,
    relative_path      = False,
    with_mask          = False)
```
