Gb_ckpt_dir = './ckpt/'
Gb_model_name = 'test.ckpt'
Gb_save_frequency = 1000
Gb_epoch = 100
Gb_learning_rate = 0.0001
Gb_jitter = 0.1

Gb_images_path = '/home/hsq/DeepLearning/data/LongWoodCutPickJpg/train/'  # /home/hsq/DeepLearning/data/car/bdd100k/images/100k/train/'  # "D:/DeepLearning/data/VOCdevkit/VOC2012/JPEGImages/"
Gb_ann_path = '/home/hsq/DeepLearning/data/LongWoodCutPickJpg/label/'  # '/home/hsq/DeepLearning/data/car/bdd100k/labels/100k/train_xml/'  # "D:/DeepLearning/data/VOCdevkit/VOC2012/Annotations/"
Gb_batch_size = 16
Gb_anchors = [125, 311, 127, 192, 212, 378, 273, 178, 324, 490, 362, 865, 404, 292, 513, 505, 639,
              727]  # [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
Gb_label = \
    ['knot']
# ['bus', 'truck', 'motor', 'car', 'train', ]
# ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
#  'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#  'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
#  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
#  'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet',
#  'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
