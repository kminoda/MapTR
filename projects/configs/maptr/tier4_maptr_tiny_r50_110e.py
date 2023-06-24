_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
roi_size = (point_cloud_range[4] - point_cloud_range[1], point_cloud_range[3] - point_cloud_range[0])

voxel_size = [0.15, 0.15, 4]

class2label = {
    'stop_line': 0,
    'divider': 1,
    'contours': 2,
    # 'ped_crossing': 3,
    'others': -1,
}

line_type_mapping = {
    # "WAY_TYPE/RELATIONS_SUBTYPE": ID
    # Road
    "line_thin:road": 1,
    "line_thick:road": 1,
    "pedestrian_marking:road": -1,
    "road_border:road": 2,
    "virtual:road": -1,

    # Crosswalk
    # "line_thin:crosswalk": 3,
    # "line_thick:crosswalk": 3,
    # "pedestrian_marking:crosswalk": 3,
    # "road_border:crosswalk": 3,
    # "virtual:crosswalk": 3,
    "line_thin:crosswalk": -1,
    "line_thick:crosswalk": -1,
    "pedestrian_marking:crosswalk": -1,
    "road_border:crosswalk": -1,
    "virtual:crosswalk": -1,

    # Walkway
    "line_thin:walkway": -1,
    "line_thick:walkway": -1,
    "pedestrian_marking:walkway": -1,
    "road_boader:walkway": -1,
    "virtual:walkway": -1,

    # Stop_line not supported yet, since some of them are not associated to relation, making it difficult to extract
    # Traffic light
    "stop_line:traffic_light": 0,

    # None
    "stop_line:none": 0,
    "road_border:none": 2,
    "curbstone:none": 2,
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing','boundary']
# fixed_ptsnum_per_line = 20
# map_classes = ['divider',]
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
# bev_h_ = 50
# bev_w_ = 50
bev_h_ = 200
bev_w_ = 100
queue_length = 1 # each sequence contains `queue_length` frames.

model = dict(
    type='MapTR',
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='MapTRHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec=50,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='MapTRPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3,5),
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', 
                      loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', 
                      weight=5),
            pc_range=point_cloud_range))))

dataset_type = 'Tier4MapDataset'
data_root = 'data/tier4_vectormap_dataset/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='Augmentation'),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='VectorizeLanelet2Map',
        roi_size=roi_size,
        line_type_mapping=line_type_mapping,
        dataroot='./data/tier4_vectormap_dataset/all_data',
        class2label=class2label,
    ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='VectorizeLanelet2Map',
        roi_size=roi_size,
        line_type_mapping=line_type_mapping,
        dataroot='./data/tier4_vectormap_dataset/valid_data',
        class2label=class2label,
    ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            # dict(
            #     type='DefaultFormatBundle3D',
            #     class_names=class_names,
            #     with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        dataroot='./data/tier4_vectormap_dataset/all_data',
        pipeline=train_pipeline,
        is_train=True,
        img_norm_cfg=img_norm_cfg,
        # data_root=data_root,
        # ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        # pipeline=train_pipeline,
        # classes=class_names,
        # modality=input_modality,
        # test_mode=False,
        # use_valid_flag=True,
        # bev_size=(bev_h_, bev_w_),
        # pc_range=point_cloud_range,
        # fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        # eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        # padding_value=-10000,
        # map_classes=map_classes,
        # queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        dataroot='./data/tier4_vectormap_dataset/valid_data',
        pipeline=test_pipeline,
        is_train=True,
        img_norm_cfg=img_norm_cfg,
        # data_root=data_root,
        # ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        # map_ann_file=data_root + 'nuscenes_map_anns_val.json',
        # pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
        # pc_range=point_cloud_range,
        # fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        # eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        # padding_value=-10000,
        # map_classes=map_classes,
        # classes=class_names, modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        dataroot='./data/tier4_vectormap_dataset/valid_data',
        pipeline=test_pipeline,
        is_train=True,
        # data_root=data_root,
        # ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        # map_ann_file=data_root + 'nuscenes_map_anns_val.json',
        # pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
        # pc_range=point_cloud_range,
        # fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        # eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        # padding_value=-10000,
        # map_classes=map_classes,
        # classes=class_names, modality=input_modality
        ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 220
# total_epochs = 50
# evaluation = dict(interval=1, pipeline=test_pipeline)
evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(interval=5)
