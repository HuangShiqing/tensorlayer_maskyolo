Gb_colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
             [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
             [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
Gb_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
             'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
             'tvmonitor']
Gb_origin_img_path = 'D:/DeepLearning/data2/VOCdevkit/VOC2012/JPEGImages/'
Gb_segment_img_path = 'D:/DeepLearning/data2/VOCdevkit/VOC2012/SegmentationClass/'
Gb_ann_path = 'D:/DeepLearning/data2/VOCdevkit/VOC2012/Annotations/'
Gb_batch_size = 16

Gb_ckpt_dir = './ckpt/'
# Gb_model_name = 'test.ckpt'
Gb_save_frequency = 499
Gb_epoch = 100
Gb_learning_rate = 0.0001
Gb_jitter = 0.1
# Gb_anchors = [125, 311, 127, 192, 212, 378, 273, 178, 324, 490, 362, 865, 404, 292, 513, 505, 639,
#               727]  # [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

